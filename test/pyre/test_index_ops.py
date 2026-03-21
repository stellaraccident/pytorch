"""Tests for gather, index, and scatter ops on the pyre host backend."""
import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TestCase


class TestEmbedding(TestCase):
    def test_basic(self):
        weight = torch.randn(100, 64).to("host:0")
        indices = torch.tensor([0, 5, 10, 50]).to("host:0")
        result = torch.ops.aten.embedding(weight, indices, -1, False, False)
        ref = torch.ops.aten.embedding(weight.cpu(), indices.cpu(), -1, False, False)
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_nn_embedding(self):
        emb = torch.nn.Embedding(100, 32)
        w = emb.weight.to("host:0")
        idx = torch.tensor([0, 99, 42]).to("host:0")
        result = torch.ops.aten.embedding(w, idx, -1, False, False)
        ref = emb(idx.cpu())
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_2d_indices(self):
        weight = torch.randn(50, 16).to("host:0")
        indices = torch.tensor([[0, 1], [2, 3]]).to("host:0")
        result = torch.ops.aten.embedding(weight, indices, -1, False, False)
        ref = torch.ops.aten.embedding(weight.cpu(), indices.cpu(), -1, False, False)
        self.assertEqual(result.cpu(), ref)
        self.assertEqual(result.shape, torch.Size([2, 2, 16]))


class TestIndexSelect(TestCase):
    def test_dim0(self):
        x = torch.randn(8, 4).to("host:0")
        idx = torch.tensor([0, 3, 5]).to("host:0")
        result = torch.index_select(x, 0, idx)
        ref = torch.index_select(x.cpu(), 0, idx.cpu())
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_dim1(self):
        x = torch.randn(4, 8).to("host:0")
        idx = torch.tensor([1, 3, 7]).to("host:0")
        result = torch.index_select(x, 1, idx)
        ref = torch.index_select(x.cpu(), 1, idx.cpu())
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)


class TestGather(TestCase):
    def test_basic(self):
        x = torch.randn(4, 5).to("host:0")
        idx = torch.tensor([[0, 1, 2], [3, 4, 0]]).to("host:0")
        result = torch.gather(x, 1, idx)
        ref = torch.gather(x.cpu(), 1, idx.cpu())
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)


class TestIndexTensor(TestCase):
    def test_1d_index(self):
        x = torch.randn(8, 4).to("host:0")
        idx = torch.tensor([0, 3, 5]).to("host:0")
        result = x[idx]
        ref = x.cpu()[idx.cpu()]
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_none_padded_index(self):
        x = torch.randn(10, 4, 8).to("host:0")
        idx = torch.tensor([0, 2, 5]).to("host:0")
        result = x[None, None, idx]
        ref = x.cpu()[None, None, idx.cpu()]
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)
        self.assertEqual(result.shape, torch.Size([1, 1, 3, 4, 8]))

    def test_2d_index(self):
        x = torch.randn(10, 4).to("host:0")
        idx = torch.tensor([[0, 1], [2, 3]]).to("host:0")
        result = x[idx]
        ref = x.cpu()[idx.cpu()]
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)


class TestScatter(TestCase):
    def test_scatter_src(self):
        x = torch.zeros(4, 5).to("host:0")
        idx = torch.tensor([[0, 1, 2], [3, 4, 0]]).to("host:0")
        src = torch.ones(2, 3).to("host:0")
        result = torch.scatter(x, 1, idx, src)
        ref = torch.scatter(torch.zeros(4, 5), 1, idx.cpu(), torch.ones(2, 3))
        self.assertEqual(result.cpu(), ref)

    def test_scatter_add(self):
        x = torch.zeros(4, 5).to("host:0")
        idx = torch.tensor([[0, 1, 2], [3, 4, 0]]).to("host:0")
        src = torch.ones(2, 3).to("host:0")
        result = x.scatter_add(1, idx, src)
        ref = torch.zeros(4, 5).scatter_add(1, idx.cpu(), torch.ones(2, 3))
        self.assertEqual(result.cpu(), ref)

    def test_scatter_inplace(self):
        x = torch.zeros(4, 5).to("host:0")
        idx = torch.tensor([[0, 1, 2]]).to("host:0")
        src = torch.ones(1, 3).to("host:0")
        x.scatter_(1, idx, src)
        ref = torch.zeros(4, 5)
        ref.scatter_(1, idx.cpu(), torch.ones(1, 3))
        self.assertEqual(x.cpu(), ref)


class TestIndexPut(TestCase):
    def test_basic(self):
        x = torch.zeros(5, 3).to("host:0")
        idx = torch.tensor([0, 2, 4]).to("host:0")
        vals = torch.ones(3, 3).to("host:0")
        result = torch.ops.aten.index_put(x, [idx], vals, False)
        ref = torch.ops.aten.index_put(torch.zeros(5, 3), [idx.cpu()],
                                        torch.ones(3, 3), False)
        self.assertEqual(result.cpu(), ref)

    def test_kv_cache_pattern(self):
        x = torch.zeros(2, 3, 8, 4).to("host:0")
        pos = torch.tensor([1, 5]).to("host:0")
        k_val = torch.randn(2, 3, 2, 4)
        x_ref = torch.zeros(2, 3, 8, 4)
        torch.ops.aten.index_put_(x, [None, None, pos], k_val.to("host:0"), False)
        torch.ops.aten.index_put_(x_ref, [None, None, pos.cpu()], k_val, False)
        self.assertEqual(x.cpu(), x_ref, atol=1e-5, rtol=1e-5)


class TestSdpa(TestCase):
    def test_basic(self):
        q = torch.randn(1, 4, 8, 16).to("host:0")
        k = torch.randn(1, 4, 8, 16).to("host:0")
        v = torch.randn(1, 4, 8, 16).to("host:0")
        result = F.scaled_dot_product_attention(q, k, v)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            ref = F.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu())
        self.assertEqual(result.cpu(), ref, atol=1e-4, rtol=1e-4)

    def test_causal(self):
        q = torch.randn(1, 4, 8, 16).to("host:0")
        k = torch.randn(1, 4, 8, 16).to("host:0")
        v = torch.randn(1, 4, 8, 16).to("host:0")
        result = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            ref = F.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu(),
                                                  is_causal=True)
        self.assertEqual(result.cpu(), ref, atol=1e-4, rtol=1e-4)

    def test_softmax(self):
        x = torch.randn(4, 8).to("host:0")
        result = torch.softmax(x, dim=-1)
        ref = torch.softmax(x.cpu(), dim=-1)
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_log_softmax(self):
        x = torch.randn(4, 8).to("host:0")
        result = torch.log_softmax(x, dim=-1)
        ref = torch.log_softmax(x.cpu(), dim=-1)
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_matmul_4d(self):
        a = torch.randn(1, 4, 8, 16).to("host:0")
        b = torch.randn(1, 4, 16, 8).to("host:0")
        result = torch.matmul(a, b)
        ref = torch.matmul(a.cpu(), b.cpu())
        self.assertEqual(result.cpu(), ref, atol=1e-4, rtol=1e-4)


class TestCreation(TestCase):
    def test_arange_int(self):
        result = torch.arange(10, device="host:0")
        self.assertEqual(result.cpu(), torch.arange(10))

    def test_arange_float(self):
        result = torch.arange(0.0, 1.0, 0.1, device="host:0")
        self.assertEqual(result.cpu(), torch.arange(0.0, 1.0, 0.1),
                         atol=1e-5, rtol=1e-5)

    def test_zeros_like(self):
        x = torch.randn(4, 4).to("host:0")
        result = torch.zeros_like(x)
        self.assertTrue((result.cpu() == 0).all())

    def test_ones_like(self):
        x = torch.randn(4, 4).to("host:0")
        result = torch.ones_like(x)
        self.assertTrue((result.cpu() == 1).all())


if __name__ == "__main__":
    run_tests()
