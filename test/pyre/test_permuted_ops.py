"""Tests for ops with permuted (non-contiguous) inputs via arg adapters."""
import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TestCase

D = "host:0"


class TestPermutedBinary(TestCase):
    """Binary ops with transposed inputs — exercises adapter template path."""

    def _check_binary(self, op, a_cpu, b_cpu, transpose_a=False, transpose_b=False):
        a = a_cpu.to(D)
        b = b_cpu.to(D)
        if transpose_a:
            a = a.t()
            a_cpu = a_cpu.t()
        if transpose_b:
            b = b.t()
            b_cpu = b_cpu.t()
        result = op(a, b)
        ref = op(a_cpu, b_cpu)
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_add_transposed_lhs(self):
        self._check_binary(torch.add, torch.randn(4, 8), torch.randn(8, 4),
                           transpose_a=True)

    def test_mul_transposed_rhs(self):
        self._check_binary(torch.mul, torch.randn(3, 5), torch.randn(5, 3),
                           transpose_b=True)

    def test_sub_both_transposed(self):
        self._check_binary(torch.sub, torch.randn(4, 6), torch.randn(4, 6),
                           transpose_a=True, transpose_b=True)

    def test_div_transposed_lhs(self):
        a = torch.randn(4, 8)
        b = torch.randn(8, 4).clamp(min=0.1)
        self._check_binary(torch.div, a, b, transpose_a=True)

    def test_add_with_alpha_transposed(self):
        a = torch.randn(4, 8).to(D).t()
        b = torch.randn(8, 4).to(D)
        result = torch.add(a, b, alpha=2.5)
        ref = torch.add(a.cpu(), b.cpu(), alpha=2.5)
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)


class TestPermutedUnary(TestCase):
    """Unary ops with transposed inputs."""

    def _check_unary(self, op, x_cpu):
        x = x_cpu.to(D).t()
        result = op(x)
        ref = op(x_cpu.t())
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_neg_transposed(self):
        self._check_unary(torch.neg, torch.randn(4, 8))

    def test_relu_transposed(self):
        self._check_unary(torch.relu, torch.randn(4, 8))

    def test_sigmoid_transposed(self):
        self._check_unary(torch.sigmoid, torch.randn(4, 8))

    def test_abs_transposed(self):
        self._check_unary(torch.abs, torch.randn(4, 8))

    def test_silu_transposed(self):
        self._check_unary(F.silu, torch.randn(4, 8))


class TestPermutedParameterized(TestCase):
    """Parameterized unary ops with transposed inputs."""

    def test_gelu_transposed(self):
        x = torch.randn(4, 8).to(D).t()
        result = F.gelu(x)
        ref = F.gelu(x.cpu())
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_hardtanh_transposed(self):
        x = (torch.randn(4, 8) * 3).to(D).t()
        result = F.hardtanh(x, min_val=-2.0, max_val=2.0)
        ref = F.hardtanh(x.cpu(), min_val=-2.0, max_val=2.0)
        self.assertEqual(result.cpu(), ref)

    def test_leaky_relu_transposed(self):
        x = torch.randn(4, 8).to(D).t()
        result = F.leaky_relu(x, negative_slope=0.2)
        ref = F.leaky_relu(x.cpu(), negative_slope=0.2)
        self.assertEqual(result.cpu(), ref)


class TestPermutedScalarBinary(TestCase):
    """Scalar binary ops with transposed inputs."""

    def test_add_scalar_transposed(self):
        x = torch.randn(4, 8).to(D).t()
        result = x + 3.0
        ref = x.cpu() + 3.0
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_mul_scalar_transposed(self):
        x = torch.randn(4, 8).to(D).t()
        result = x * 0.5
        ref = x.cpu() * 0.5
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)


class TestPermutedComparison(TestCase):
    """Comparison ops with transposed inputs."""

    def test_gt_transposed(self):
        a = torch.randn(4, 8).to(D).t()
        b = torch.randn(8, 4).to(D)
        result = a > b
        ref = a.cpu() > b.cpu()
        self.assertEqual(result.cpu(), ref)

    def test_eq_scalar_transposed(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 1.0]]).to(D).t()
        result = x == 1.0
        ref = x.cpu() == 1.0
        self.assertEqual(result.cpu(), ref)


class TestPermutedMm(TestCase):
    """mm/bmm with transposed inputs."""

    def test_mm_rhs_transposed(self):
        a = torch.randn(2, 3).to(D)
        w = torch.randn(4, 3).to(D)
        result = torch.mm(a, w.t())
        ref = torch.mm(a.cpu(), w.cpu().t())
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_mm_lhs_transposed(self):
        a = torch.randn(3, 2).to(D)
        b = torch.randn(3, 4).to(D)
        result = torch.mm(a.t(), b)
        ref = torch.mm(a.cpu().t(), b.cpu())
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_linear(self):
        x = torch.randn(1, 1, 64).to(D)
        w = torch.randn(192, 64).to(D)
        result = F.linear(x, w)
        ref = F.linear(x.cpu(), w.cpu())
        self.assertEqual(result.cpu(), ref, atol=1e-4, rtol=1e-4)


class TestPermuted3D(TestCase):
    """3D permutation (not just 2D transpose)."""

    def test_add_3d_permuted(self):
        x = torch.randn(2, 3, 4).to(D).permute(2, 0, 1)
        y = torch.randn(4, 2, 3).to(D)
        result = x + y
        ref = x.cpu() + y.cpu()
        self.assertEqual(result.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_neg_3d_permuted(self):
        x = torch.randn(2, 3, 4).to(D).permute(2, 0, 1)
        result = -x
        ref = -x.cpu()
        self.assertEqual(result.cpu(), ref)


if __name__ == "__main__":
    run_tests()
