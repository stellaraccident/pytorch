"""Tests for Epic 3 op breadth: unary, binary, scalar, comparison, reduction, etc."""

import unittest
import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TestCase


class TestPureUnaryOps(TestCase):
    def _check(self, fn, x_cpu, atol=1e-5):
        x = x_cpu.to("host:0")
        y = fn(x)
        ref = fn(x_cpu)
        self.assertTrue(torch.allclose(y.cpu(), ref, atol=atol),
                        msg=f"mismatch for {fn}")

    def test_silu(self):
        self._check(F.silu, torch.randn(8))

    def test_sigmoid(self):
        self._check(torch.sigmoid, torch.randn(8))

    def test_tanh(self):
        self._check(torch.tanh, torch.randn(8))

    def test_rsqrt(self):
        self._check(torch.rsqrt, torch.rand(8) + 0.1)

    def test_exp(self):
        self._check(torch.exp, torch.randn(8))

    def test_log(self):
        self._check(torch.log, torch.rand(8) + 0.1)

    def test_sqrt(self):
        self._check(torch.sqrt, torch.rand(8) + 0.1)

    def test_sin(self):
        self._check(torch.sin, torch.randn(8))

    def test_cos(self):
        self._check(torch.cos, torch.randn(8))

    def test_ceil(self):
        self._check(torch.ceil, torch.randn(8))

    def test_floor(self):
        self._check(torch.floor, torch.randn(8))

    def test_round(self):
        self._check(torch.round, torch.randn(8))

    def test_reciprocal(self):
        self._check(torch.reciprocal, torch.randn(8).abs() + 0.1)

    def test_erf(self):
        self._check(torch.erf, torch.randn(8))

    def test_sign(self):
        self._check(torch.sign, torch.randn(8))

    def test_bitwise_not(self):
        x = torch.tensor([1, 0, -1, 42], dtype=torch.int32)
        y = torch.bitwise_not(x.to("host:0"))
        self.assertEqual(y.cpu(), torch.bitwise_not(x))

    def test_logical_not(self):
        x = torch.tensor([True, False, True, False])
        y = torch.logical_not(x.to("host:0"))
        self.assertEqual(y.cpu(), torch.logical_not(x))


class TestParameterizedUnaryOps(TestCase):
    def test_gelu(self):
        x = torch.randn(8)
        y = F.gelu(x.to("host:0"))
        self.assertTrue(torch.allclose(y.cpu(), F.gelu(x), atol=1e-5))

    def test_hardtanh(self):
        x = torch.randn(8)
        y = F.hardtanh(x.to("host:0"))
        self.assertTrue(torch.allclose(y.cpu(), F.hardtanh(x), atol=1e-5))

    def test_hardtanh_custom_bounds(self):
        x = torch.randn(8)
        y = F.hardtanh(x.to("host:0"), min_val=-2.0, max_val=2.0)
        self.assertTrue(torch.allclose(y.cpu(), F.hardtanh(x, -2.0, 2.0), atol=1e-5))

    def test_leaky_relu(self):
        x = torch.randn(8)
        y = F.leaky_relu(x.to("host:0"))
        self.assertTrue(torch.allclose(y.cpu(), F.leaky_relu(x), atol=1e-5))

    def test_leaky_relu_custom_slope(self):
        x = torch.randn(8)
        y = F.leaky_relu(x.to("host:0"), negative_slope=0.2)
        self.assertTrue(torch.allclose(y.cpu(), F.leaky_relu(x, 0.2), atol=1e-5))

    def test_elu_custom_alpha(self):
        x = torch.randn(8)
        y = F.elu(x.to("host:0"), alpha=2.0)
        self.assertTrue(torch.allclose(y.cpu(), F.elu(x, alpha=2.0), atol=1e-5))

    def test_elu(self):
        x = torch.randn(8)
        y = F.elu(x.to("host:0"))
        self.assertTrue(torch.allclose(y.cpu(), F.elu(x), atol=1e-5))


class TestBinaryOps(TestCase):
    def _check(self, fn, a_cpu, b_cpu, atol=1e-5):
        y = fn(a_cpu.to("host:0"), b_cpu.to("host:0"))
        ref = fn(a_cpu, b_cpu)
        self.assertTrue(torch.allclose(y.cpu(), ref, atol=atol))

    def test_maximum(self):
        self._check(torch.maximum, torch.randn(8), torch.randn(8))

    def test_minimum(self):
        self._check(torch.minimum, torch.randn(8), torch.randn(8))

    def test_pow(self):
        self._check(torch.pow, torch.rand(8) + 0.1, torch.rand(8) + 0.1)

    def test_atan2(self):
        self._check(torch.atan2, torch.randn(8), torch.randn(8))

    def test_bitwise_and(self):
        a = torch.tensor([0xFF, 0x0F], dtype=torch.int32)
        b = torch.tensor([0xF0, 0x0F], dtype=torch.int32)
        y = torch.bitwise_and(a.to("host:0"), b.to("host:0"))
        self.assertEqual(y.cpu(), torch.bitwise_and(a, b))

    def test_remainder_float(self):
        self._check(torch.remainder, torch.randn(8), torch.rand(8) + 0.5)

    def test_fmod_float(self):
        self._check(torch.fmod, torch.randn(8), torch.rand(8) + 0.5)


class TestScalarBinaryOps(TestCase):
    def test_add_scalar(self):
        x = torch.randn(8)
        y = torch.ops.aten.add.Scalar(x.to("host:0"), 2.5)
        self.assertTrue(torch.allclose(y.cpu(), x + 2.5, atol=1e-5))

    def test_mul_scalar(self):
        x = torch.randn(8)
        y = torch.ops.aten.mul.Scalar(x.to("host:0"), 3.0)
        self.assertTrue(torch.allclose(y.cpu(), x * 3.0, atol=1e-5))

    def test_div_scalar(self):
        x = torch.randn(8)
        y = torch.ops.aten.div.Scalar(x.to("host:0"), 2.0)
        self.assertTrue(torch.allclose(y.cpu(), x / 2.0, atol=1e-5))

    def test_pow_scalar(self):
        x = torch.rand(8) + 0.1
        y = torch.ops.aten.pow.Tensor_Scalar(x.to("host:0"), 2.0)
        self.assertTrue(torch.allclose(y.cpu(), x.pow(2.0), atol=1e-5))


class TestComparisonOps(TestCase):
    def test_eq_tensor(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 0.0, 3.0])
        y = torch.eq(a.to("host:0"), b.to("host:0"))
        self.assertEqual(y.cpu(), torch.eq(a, b))

    def test_lt_tensor(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([2.0, 2.0, 1.0])
        y = torch.lt(a.to("host:0"), b.to("host:0"))
        self.assertEqual(y.cpu(), torch.lt(a, b))

    def test_ge_scalar(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        y = torch.ops.aten.ge.Scalar(a.to("host:0"), 2.0)
        self.assertEqual(y.cpu(), torch.ge(a, 2.0))

    def test_ne_scalar(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        y = torch.ops.aten.ne.Scalar(a.to("host:0"), 2.0)
        self.assertEqual(y.cpu(), torch.ne(a, 2.0))


class TestReductionOps(TestCase):
    def test_sum_dim(self):
        x = torch.randn(3, 4)
        y = x.to("host:0").sum(dim=1)
        self.assertTrue(torch.allclose(y.cpu(), x.sum(dim=1), atol=1e-5))

    def test_sum_keepdim(self):
        x = torch.randn(3, 4)
        y = x.to("host:0").sum(dim=0, keepdim=True)
        self.assertTrue(torch.allclose(y.cpu(), x.sum(dim=0, keepdim=True), atol=1e-5))

    def test_mean_dim(self):
        x = torch.randn(3, 4)
        y = x.to("host:0").mean(dim=1)
        self.assertTrue(torch.allclose(y.cpu(), x.mean(dim=1), atol=1e-5))

    def test_amax(self):
        x = torch.randn(3, 4)
        y = x.to("host:0").amax(dim=1)
        self.assertTrue(torch.allclose(y.cpu(), x.amax(dim=1), atol=1e-5))

    def test_amin(self):
        x = torch.randn(3, 4)
        y = x.to("host:0").amin(dim=0)
        self.assertTrue(torch.allclose(y.cpu(), x.amin(dim=0), atol=1e-5))

    def test_prod(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = x.to("host:0").prod(dim=1)
        self.assertTrue(torch.allclose(y.cpu(), x.prod(dim=1), atol=1e-4))

    def test_prod_keepdim(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = x.to("host:0").prod(dim=0, keepdim=True)
        self.assertTrue(torch.allclose(y.cpu(), x.prod(dim=0, keepdim=True), atol=1e-4))


class TestTypeCast(TestCase):
    def test_f32_to_bf16(self):
        x = torch.randn(8)
        y = x.to("host:0").to(torch.bfloat16)
        self.assertEqual(y.cpu(), x.to(torch.bfloat16))

    @unittest.skip("TODO: f32->f64 cast produces wrong values (pyre-workspace-o8w)")
    def test_f32_to_f64(self):
        x = torch.randn(8)
        y = x.to("host:0").to(torch.float64)
        self.assertTrue(torch.allclose(y.cpu(), x.to(torch.float64), atol=1e-5))


class TestBmm(TestCase):
    def test_basic(self):
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 4, 5)
        y = torch.bmm(a.to("host:0"), b.to("host:0"))
        self.assertTrue(torch.allclose(y.cpu(), torch.bmm(a, b), atol=1e-4))


class TestWhere(TestCase):
    def test_basic(self):
        cond = torch.tensor([True, False, True, False])
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        other = torch.tensor([10.0, 20.0, 30.0, 40.0])
        y = torch.where(cond.to("host:0"), x.to("host:0"), other.to("host:0"))
        self.assertEqual(y.cpu(), torch.where(cond, x, other))


class TestViewOps(TestCase):
    def test_select(self):
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        y = x.to("host:0").select(0, 1)
        self.assertEqual(y.cpu(), x.select(0, 1))

    def test_narrow(self):
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        y = x.to("host:0").narrow(1, 1, 2)
        self.assertEqual(y.cpu(), x.narrow(1, 1, 2))

    def test_transpose(self):
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        y = x.to("host:0").transpose(0, 1)
        self.assertEqual(y.cpu(), x.transpose(0, 1))

    def test_split(self):
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        parts = x.to("host:0").split(2, dim=1)
        refs = x.split(2, dim=1)
        for p, r in zip(parts, refs):
            self.assertEqual(p.cpu(), r)

    def test_chunk(self):
        x = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        parts = x.to("host:0").chunk(3, dim=0)
        refs = x.chunk(3, dim=0)
        for p, r in zip(parts, refs):
            self.assertEqual(p.cpu(), r)

    def test_clone(self):
        x = torch.randn(4, 4)
        y = x.to("host:0").clone()
        self.assertEqual(y.cpu(), x)


class TestCat(TestCase):
    def test_two_inputs(self):
        a = torch.randn(2, 3)
        b = torch.randn(2, 4)
        y = torch.cat([a.to("host:0"), b.to("host:0")], dim=1)
        self.assertTrue(torch.allclose(y.cpu(), torch.cat([a, b], dim=1), atol=1e-5))

    def test_three_inputs(self):
        a = torch.randn(2, 3)
        b = torch.randn(2, 4)
        c = torch.randn(2, 5)
        y = torch.cat([a.to("host:0"), b.to("host:0"), c.to("host:0")], dim=1)
        self.assertTrue(torch.allclose(y.cpu(), torch.cat([a, b, c], dim=1), atol=1e-5))

    def test_dim0(self):
        a = torch.randn(2, 4)
        b = torch.randn(3, 4)
        y = torch.cat([a.to("host:0"), b.to("host:0")], dim=0)
        self.assertTrue(torch.allclose(y.cpu(), torch.cat([a, b], dim=0), atol=1e-5))


if __name__ == "__main__":
    run_tests()
