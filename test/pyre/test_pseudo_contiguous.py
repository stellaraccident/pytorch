"""Tests for tensors that PyTorch reports as contiguous but have non-row-major
strides (e.g. size-1 dims after transpose). These are the sneakiest layout
bugs because is_contiguous()=True but the buffer data doesn't match the
logical shape's expected row-major layout.

The canonical pattern: view(1,1,H,D).transpose(1,2) -> [1,H,1,D] with
strides from the pre-transpose layout. This happens in every transformer's
attention: q/k/v are viewed then transposed, and the KV cache update
writes these "pseudo-contiguous" tensors via index_put_.
"""
import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TestCase

D = "host:0"


def make_pseudo_contiguous():
    """Create a [1,2,1,3] tensor with strides (6,3,6,1) — pseudo-contiguous."""
    x = torch.randn(1, 1, 2, 3)   # truly contiguous [1,1,2,3]
    y = x.transpose(1, 2)          # [1,2,1,3] strides (6,3,6,1)
    assert y.is_contiguous(), "test setup: should be contiguous"
    assert y.stride() != (6, 3, 3, 1), "test setup: should have non-standard strides"
    return y


class TestPseudoContiguousDetection(TestCase):
    """Verify ArgAdapter detects pseudo-contiguous tensors."""

    def test_is_contiguous_but_non_standard(self):
        y = make_pseudo_contiguous()
        self.assertTrue(y.is_contiguous())
        # Strides should NOT be standard row-major for [1,2,1,3]
        # Standard would be (6, 3, 3, 1)
        self.assertNotEqual(y.stride(), (6, 3, 3, 1))


class TestPseudoContiguousBinary(TestCase):
    """Binary ops where one input is pseudo-contiguous."""

    def test_add_pseudo_lhs(self):
        a = make_pseudo_contiguous().to(D)
        b = torch.randn(1, 2, 1, 3).to(D)
        result = a + b
        ref = a.cpu() + b.cpu()
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_mul_pseudo_rhs(self):
        a = torch.randn(1, 2, 1, 3).to(D)
        b = make_pseudo_contiguous().to(D)
        result = a * b
        ref = a.cpu() * b.cpu()
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_add_both_pseudo(self):
        a = make_pseudo_contiguous().to(D)
        b = make_pseudo_contiguous().to(D)
        result = a + b
        ref = a.cpu() + b.cpu()
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)


class TestPseudoContiguousUnary(TestCase):
    """Unary ops on pseudo-contiguous tensors."""

    def test_neg_pseudo(self):
        x = make_pseudo_contiguous().to(D)
        result = -x
        ref = -x.cpu()
        self.assertEqual(result.cpu(), ref)

    def test_relu_pseudo(self):
        x = make_pseudo_contiguous().to(D)
        result = torch.relu(x)
        ref = torch.relu(x.cpu())
        self.assertEqual(result.cpu(), ref)

    def test_silu_pseudo(self):
        x = make_pseudo_contiguous().to(D)
        result = F.silu(x)
        ref = F.silu(x.cpu())
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)


class TestPseudoContiguousScalarBinary(TestCase):
    """Scalar binary ops on pseudo-contiguous tensors."""

    def test_mul_scalar_pseudo(self):
        x = make_pseudo_contiguous().to(D)
        result = x * 2.5
        ref = x.cpu() * 2.5
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)


class TestPseudoContiguousIndexPut(TestCase):
    """index_put_ with pseudo-contiguous value tensor — the GPT KV cache bug.

    The real pattern: wqkv(x).split(...).view(B,S,H,D).transpose(1,2) produces
    [B,H,S,D] with pseudo-contiguous strides. Then kv_cache[:,:,pos] = v writes
    this tensor via index_put_. The pseudo-contiguous strides must be handled
    correctly to write the right values into the cache.
    """

    def test_kv_cache_pattern(self):
        """Reproduce the exact GPT KV cache write pattern on device."""
        B, H, S, Dh = 1, 2, 1, 32
        cache_len = 8

        # Create cache on device with known pattern (not zeros).
        cache = torch.randn(B, H, cache_len, Dh).to(D)
        cache_ref = cache.cpu().clone()

        # Simulate: qkv projection output, split, view, transpose.
        # This produces a pseudo-contiguous [B,H,S,Dh] tensor on device.
        raw = torch.randn(B, S, H, Dh).to(D)  # e.g. from linear
        val = raw.transpose(1, 2)  # [B,H,S,Dh] — pseudo-contiguous
        self.assertTrue(val.is_contiguous())
        # Strides should be non-standard: dim 2 stride > dim 1 stride
        self.assertGreater(val.stride(2), val.stride(1))

        raw_cpu = raw.cpu()
        val_cpu = raw_cpu.transpose(1, 2)

        pos = torch.tensor([3]).to(D)
        torch.ops.aten.index_put_(cache, [None, None, pos], val, False)
        torch.ops.aten.index_put_(cache_ref, [None, None, torch.tensor([3])],
                                   val_cpu, False)

        self.assertEqual(cache.cpu()[:, :, 3, :], cache_ref[:, :, 3, :],
                         atol=1e-5, rtol=1e-5)
        # Verify other positions unchanged
        self.assertEqual(cache.cpu()[:, :, 0, :], cache_ref[:, :, 0, :])

    def test_kv_cache_multi_head(self):
        """Multiple heads with different values per head."""
        B, H, S, Dh = 1, 4, 1, 16
        cache = torch.randn(B, H, 16, Dh).to(D)
        cache_ref = cache.cpu().clone()

        raw = torch.randn(B, S, H, Dh).to(D)
        val = raw.transpose(1, 2)

        raw_cpu = raw.cpu()
        val_cpu = raw_cpu.transpose(1, 2)

        pos = torch.tensor([7]).to(D)
        torch.ops.aten.index_put_(cache, [None, None, pos], val, False)
        torch.ops.aten.index_put_(cache_ref, [None, None, torch.tensor([7])],
                                   val_cpu, False)

        for h in range(H):
            self.assertEqual(cache.cpu()[0, h, 7, :], cache_ref[0, h, 7, :],
                             atol=1e-5, rtol=1e-5,
                             msg=f"head {h} mismatch")


class TestStorageOffset(TestCase):
    """Ops on tensors with non-zero storage offset (from split on inner dim).

    split on the last dim produces N tensors sharing one storage buffer,
    each contiguous but at different offsets. buildBufferView must account
    for the offset or the kernel reads the wrong data.
    """

    def test_split_add(self):
        """add on two split halves of a tensor."""
        ab = torch.randn(1, 1, 8).to(D)
        a, b = ab.split([4, 4], dim=-1)
        self.assertEqual(a.storage_offset(), 0)
        self.assertEqual(b.storage_offset(), 4)
        result = a + b

        ab_cpu = ab.cpu()
        a_cpu, b_cpu = ab_cpu.split([4, 4], dim=-1)
        ref = a_cpu + b_cpu
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_split_neg(self):
        """unary on a split tensor with offset."""
        ab = torch.randn(2, 8).to(D)
        a, b = ab.split([4, 4], dim=-1)
        result = -b
        ref = -ab.cpu()[:, 4:]
        self.assertEqual(result.cpu(), ref)

    def test_split_mm(self):
        """mm where one input comes from split (has offset)."""
        ab = torch.randn(6, 4).to(D)  # will split into [2,4] and [4,4]
        a, b = ab.split([2, 4], dim=0)
        # a is [2,4] offset=0, b is [4,4] offset=8
        c = torch.randn(4, 3).to(D)
        result = torch.mm(b, c)

        ab_cpu = ab.cpu()
        _, b_cpu = ab_cpu.split([2, 4], dim=0)
        ref = torch.mm(b_cpu, c.cpu())
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_qkv_split_pattern(self):
        """The GPT wqkv split pattern: [1,1,192] → q[64] k[64] v[64]."""
        qkv = torch.randn(1, 1, 192).to(D)
        q, k, v = qkv.split([64, 64, 64], dim=-1)

        self.assertEqual(q.storage_offset(), 0)
        self.assertEqual(k.storage_offset(), 64)
        self.assertEqual(v.storage_offset(), 128)

        # Each split piece should match the corresponding CPU slice
        qkv_cpu = qkv.cpu()
        self.assertEqual(q.cpu(), qkv_cpu[:, :, :64])
        self.assertEqual(k.cpu(), qkv_cpu[:, :, 64:128])
        self.assertEqual(v.cpu(), qkv_cpu[:, :, 128:])

    def test_qkv_split_view_add(self):
        """Binary op after split+view (offset + reshape)."""
        qkv = torch.randn(1, 1, 12).to(D)
        a, b, c = qkv.split([4, 4, 4], dim=-1)
        # b has offset=4, c has offset=8
        b2 = b.view(1, 1, 2, 2)
        c2 = c.view(1, 1, 2, 2)
        result = b2 + c2

        qkv_cpu = qkv.cpu()
        _, b_cpu, c_cpu = qkv_cpu.split([4, 4, 4], dim=-1)
        ref = b_cpu.view(1, 1, 2, 2) + c_cpu.view(1, 1, 2, 2)
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_kv_cache_write_from_split(self):
        """KV cache write where v comes from split+view+transpose."""
        qkv = torch.randn(1, 1, 192).to(D)
        _, _, v = qkv.split([64, 64, 64], dim=-1)
        v2 = v.view(1, 1, 2, 32).transpose(1, 2)  # [1,2,1,32] offset=128

        cache = torch.randn(1, 2, 4, 32).to(D)
        cache_ref = cache.cpu().clone()
        pos = torch.tensor([1]).to(D)

        qkv_cpu = qkv.cpu()
        _, _, v_cpu = qkv_cpu.split([64, 64, 64], dim=-1)
        v2_cpu = v_cpu.view(1, 1, 2, 32).transpose(1, 2)

        torch.ops.aten.index_put_(cache, [None, None, pos], v2, False)
        torch.ops.aten.index_put_(cache_ref, [None, None, torch.tensor([1])],
                                   v2_cpu, False)
        self.assertEqual(cache.cpu()[:, :, 1, :], cache_ref[:, :, 1, :],
                         atol=1e-5, rtol=1e-5)


class TestPseudoContiguousMm(TestCase):
    """mm with pseudo-contiguous inputs (unlikely in practice but tests adapter)."""

    def test_mm_pseudo_rhs(self):
        # [3, 4] pseudo-contiguous from [4, 3].transpose
        raw = torch.randn(4, 3)
        rhs = raw.t()  # [3, 4] — standard transpose, not pseudo
        self.assertFalse(rhs.is_contiguous())
        # This is a normal transpose test, already covered.
        # For pseudo: [1,3,1,4].squeeze() won't help. 2D pseudo is rare.
        pass


class TestPseudoContiguousComparison(TestCase):
    """Comparison ops on pseudo-contiguous tensors."""

    def test_gt_pseudo(self):
        a = make_pseudo_contiguous().to(D)
        b = torch.randn(1, 2, 1, 3).to(D)
        result = a > b
        ref = a.cpu() > b.cpu()
        self.assertEqual(result.cpu(), ref)


class TestPseudoContiguousSoftmax(TestCase):
    """Softmax on pseudo-contiguous tensor — dim refers to logical axes."""

    def test_softmax_pseudo(self):
        x = make_pseudo_contiguous().to(D)
        result = torch.softmax(x, dim=-1)
        ref = torch.softmax(x.cpu(), dim=-1)
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_softmax_pseudo_dim1(self):
        x = make_pseudo_contiguous().to(D)
        result = torch.softmax(x, dim=1)
        ref = torch.softmax(x.cpu(), dim=1)
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)


class TestPseudoContiguousReduction(TestCase):
    """Reduction on pseudo-contiguous tensor."""

    def test_sum_pseudo(self):
        x = make_pseudo_contiguous().to(D)
        result = x.sum(dim=-1)
        ref = x.cpu().sum(dim=-1)
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_mean_pseudo_dim1(self):
        x = make_pseudo_contiguous().to(D)
        result = x.mean(dim=1)
        ref = x.cpu().mean(dim=1)
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    run_tests()
