"""Tests for compiled kernel dispatch."""
import torch
from torch.testing._internal.common_utils import run_tests, TestCase


DEVICE = "host:0"


class TestBinaryOps(TestCase):
    """Binary elementwise ops: add, sub, mul, div."""

    def _check(self, op, a, b, expected, **kwargs):
        result = op(a, b, **kwargs)
        self.assertEqual(result.cpu(), expected, atol=1e-5, rtol=1e-5)

    def test_add_f32(self):
        a = torch.randn(4, 4, device=DEVICE)
        b = torch.randn(4, 4, device=DEVICE)
        self._check(torch.add, a, b, a.cpu() + b.cpu())

    def test_add_f32_1d(self):
        a = torch.randn(16, device=DEVICE)
        b = torch.randn(16, device=DEVICE)
        self._check(torch.add, a, b, a.cpu() + b.cpu())

    def test_add_f32_3d(self):
        a = torch.randn(2, 3, 4, device=DEVICE)
        b = torch.randn(2, 3, 4, device=DEVICE)
        self._check(torch.add, a, b, a.cpu() + b.cpu())

    def test_add_alpha(self):
        a = torch.randn(4, 4, device=DEVICE)
        b = torch.randn(4, 4, device=DEVICE)
        self._check(torch.add, a, b, a.cpu() + 2.0 * b.cpu(), alpha=2.0)

    def test_add_alpha_negative(self):
        a = torch.randn(4, 4, device=DEVICE)
        b = torch.randn(4, 4, device=DEVICE)
        self._check(torch.add, a, b, a.cpu() + (-3.0) * b.cpu(), alpha=-3.0)

    def test_add_i32(self):
        a = torch.randint(0, 100, (4, 4), dtype=torch.int32, device=DEVICE)
        b = torch.randint(0, 100, (4, 4), dtype=torch.int32, device=DEVICE)
        self._check(torch.add, a, b, a.cpu() + b.cpu())

    def test_sub_f32(self):
        a = torch.randn(4, 4, device=DEVICE)
        b = torch.randn(4, 4, device=DEVICE)
        self._check(torch.sub, a, b, a.cpu() - b.cpu())

    def test_sub_alpha(self):
        a = torch.randn(4, 4, device=DEVICE)
        b = torch.randn(4, 4, device=DEVICE)
        self._check(torch.sub, a, b, a.cpu() - 2.0 * b.cpu(), alpha=2.0)

    def test_mul_f32(self):
        a = torch.randn(4, 4, device=DEVICE)
        b = torch.randn(4, 4, device=DEVICE)
        self._check(torch.mul, a, b, a.cpu() * b.cpu())

    def test_div_f32(self):
        a = torch.randn(4, 4, device=DEVICE)
        b_cpu = torch.randn(4, 4).abs() + 0.1
        b = b_cpu.to(DEVICE)
        self._check(torch.div, a, b, a.cpu() / b_cpu)

    def test_sequential_cache_reuse(self):
        """Multiple dispatches reuse cached kernels."""
        for _ in range(5):
            a = torch.randn(4, 4, device=DEVICE)
            b = torch.randn(4, 4, device=DEVICE)
            r = torch.add(a, b)
            self.assertEqual(r.cpu(), a.cpu() + b.cpu())


class TestUnaryOps(TestCase):
    """Unary elementwise ops: neg, relu."""

    def test_neg_f32(self):
        a = torch.randn(4, 4, device=DEVICE)
        self.assertEqual(torch.neg(a).cpu(), -a.cpu(), atol=1e-5, rtol=1e-5)

    def test_neg_f32_1d(self):
        a = torch.randn(16, device=DEVICE)
        self.assertEqual(torch.neg(a).cpu(), -a.cpu(), atol=1e-5, rtol=1e-5)

    def test_relu_f32(self):
        a = torch.randn(4, 4, device=DEVICE)
        self.assertEqual(torch.relu(a).cpu(), torch.relu(a.cpu()), atol=1e-5, rtol=1e-5)

    def test_relu_mixed_sign(self):
        a = torch.tensor([-1.0, 0.0, 1.0, -0.5, 2.0], device=DEVICE)
        expected = torch.tensor([0.0, 0.0, 1.0, 0.0, 2.0])
        self.assertEqual(torch.relu(a).cpu(), expected)


class TestBroadcasting(TestCase):
    """Broadcasting tests — currently skipped pending static-dim support."""

    def test_add_broadcast_row(self):
        a = torch.randn(4, 4, device=DEVICE)
        b = torch.randn(1, 4, device=DEVICE)
        self.assertEqual(torch.add(a, b).cpu(), a.cpu() + b.cpu())

    def test_add_broadcast_col(self):
        a = torch.randn(4, 4, device=DEVICE)
        b = torch.randn(4, 1, device=DEVICE)
        self.assertEqual(torch.add(a, b).cpu(), a.cpu() + b.cpu())


class TestAsyncPipeline(TestCase):
    """Chained operations execute correctly (fences chain)."""

    def test_chained_add_relu_mul(self):
        a = torch.randn(4, 4, device=DEVICE)
        b = torch.randn(4, 4, device=DEVICE)
        c = torch.randn(4, 4, device=DEVICE)
        result = torch.mul(torch.relu(torch.add(a, b)), c)
        expected = torch.mul(torch.relu(a.cpu() + b.cpu()), c.cpu())
        self.assertEqual(result.cpu(), expected, atol=1e-5, rtol=1e-5)

    def test_chained_sub_neg(self):
        a = torch.randn(4, 4, device=DEVICE)
        b = torch.randn(4, 4, device=DEVICE)
        result = torch.neg(torch.sub(a, b))
        expected = -(a.cpu() - b.cpu())
        self.assertEqual(result.cpu(), expected, atol=1e-5, rtol=1e-5)


class TestPermutedOps(TestCase):
    """Fused axis permutation: compiler sees through torch.aten.permute."""

    def test_add_transposed_lhs(self):
        a = torch.randn(4, 8, device=DEVICE).t()  # [8,4] logical
        b = torch.randn(8, 4, device=DEVICE)
        result = a + b
        expected = a.cpu() + b.cpu()
        self.assertEqual(result.cpu(), expected, atol=1e-5, rtol=1e-5)

    def test_add_both_transposed(self):
        a = torch.randn(4, 8, device=DEVICE).t()
        b = torch.randn(4, 8, device=DEVICE).t()
        result = a + b
        expected = a.cpu() + b.cpu()
        self.assertEqual(result.cpu(), expected, atol=1e-5, rtol=1e-5)

    def test_neg_transposed(self):
        x = torch.randn(4, 8, device=DEVICE).t()
        result = -x
        expected = -x.cpu()
        self.assertEqual(result.cpu(), expected, atol=1e-5, rtol=1e-5)

    def test_addmm_transposed_weight(self):
        bias = torch.randn(10, device=DEVICE)
        mat1 = torch.randn(4, 20, device=DEVICE)
        weight = torch.randn(10, 20, device=DEVICE)
        mat2 = weight.t()  # [20,10] logical, [10,20] physical
        out = torch.addmm(bias, mat1, mat2)
        expected = torch.addmm(bias.cpu(), mat1.cpu(), mat2.cpu())
        self.assertEqual(out.cpu(), expected, atol=1e-5, rtol=1e-5)

    def test_3d_permuted(self):
        x = torch.randn(2, 3, 4, device=DEVICE).permute(2, 0, 1)
        y = torch.randn(4, 2, 3, device=DEVICE)
        result = x + y
        expected = x.cpu() + y.cpu()
        self.assertEqual(result.cpu(), expected, atol=1e-5, rtol=1e-5)

    def test_non_permutation_falls_back(self):
        # Sliced tensor is not a permutation — forced contiguous
        x = torch.randn(8, 8, device=DEVICE)[::2]  # stride gap
        y = torch.randn(4, 8, device=DEVICE)
        result = x + y
        expected = x.cpu() + y.cpu()
        self.assertEqual(result.cpu(), expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    run_tests()
