"""Tests for in-place op variants on the pyre host backend."""
import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TestCase


class TestInplaceOps(TestCase):
    device = "host:0"

    def _host(self, *args, dtype=torch.float32):
        return tuple(
            torch.randn(4, 4, dtype=dtype).to(self.device) if a is None else a
            for a in args
        )

    # --- Binary tensor-tensor in-place ---

    def test_add_inplace(self):
        x_cpu = torch.randn(4, 4)
        y_cpu = torch.randn(4, 4)
        x = x_cpu.to(self.device)
        y = y_cpu.to(self.device)
        x.add_(y)
        self.assertEqual(x.cpu(), x_cpu + y_cpu)

    def test_add_inplace_alpha(self):
        x_cpu = torch.randn(4, 4)
        y_cpu = torch.randn(4, 4)
        x = x_cpu.to(self.device)
        y = y_cpu.to(self.device)
        x.add_(y, alpha=2.0)
        self.assertEqual(x.cpu(), x_cpu + 2.0 * y_cpu)

    def test_sub_inplace(self):
        x_cpu = torch.randn(4, 4)
        y_cpu = torch.randn(4, 4)
        x = x_cpu.to(self.device)
        y = y_cpu.to(self.device)
        x.sub_(y)
        self.assertEqual(x.cpu(), x_cpu - y_cpu)

    def test_mul_inplace(self):
        x_cpu = torch.randn(4, 4)
        y_cpu = torch.randn(4, 4)
        x = x_cpu.to(self.device)
        y = y_cpu.to(self.device)
        x.mul_(y)
        self.assertEqual(x.cpu(), x_cpu * y_cpu)

    def test_div_inplace(self):
        x_cpu = torch.randn(4, 4)
        y_cpu = torch.randn(4, 4).clamp(min=0.1)
        x = x_cpu.to(self.device)
        y = y_cpu.to(self.device)
        x.div_(y)
        self.assertEqual(x.cpu(), x_cpu / y_cpu)

    def test_pow_inplace(self):
        x_cpu = torch.rand(4, 4) + 0.5
        y_cpu = torch.rand(4, 4) + 0.5
        x = x_cpu.to(self.device)
        y = y_cpu.to(self.device)
        x.pow_(y)
        self.assertEqual(x.cpu(), x_cpu.pow(y_cpu), atol=1e-5, rtol=1e-5)

    def test_atan2_inplace(self):
        x_cpu = torch.randn(4, 4)
        y_cpu = torch.randn(4, 4)
        x = x_cpu.to(self.device)
        y = y_cpu.to(self.device)
        x.atan2_(y)
        self.assertEqual(x.cpu(), torch.atan2(x_cpu, y_cpu))

    # --- Unary in-place ---

    def test_neg_inplace(self):
        x_cpu = torch.randn(4, 4)
        x = x_cpu.to(self.device)
        x.neg_()
        self.assertEqual(x.cpu(), -x_cpu)

    def test_relu_inplace(self):
        x_cpu = torch.randn(4, 4)
        x = x_cpu.to(self.device)
        x.relu_()
        self.assertEqual(x.cpu(), x_cpu.relu())

    def test_abs_inplace(self):
        x_cpu = torch.randn(4, 4)
        x = x_cpu.to(self.device)
        x.abs_()
        self.assertEqual(x.cpu(), x_cpu.abs())

    def test_sigmoid_inplace(self):
        x_cpu = torch.randn(4, 4)
        x = x_cpu.to(self.device)
        x.sigmoid_()
        self.assertEqual(x.cpu(), x_cpu.sigmoid())

    def test_tanh_inplace(self):
        x_cpu = torch.randn(4, 4)
        x = x_cpu.to(self.device)
        x.tanh_()
        self.assertEqual(x.cpu(), x_cpu.tanh())

    def test_exp_inplace(self):
        x_cpu = torch.randn(4, 4)
        x = x_cpu.to(self.device)
        x.exp_()
        self.assertEqual(x.cpu(), x_cpu.exp())

    def test_sqrt_inplace(self):
        x_cpu = torch.rand(4, 4) + 0.1
        x = x_cpu.to(self.device)
        x.sqrt_()
        self.assertEqual(x.cpu(), x_cpu.sqrt())

    def test_cos_inplace(self):
        x_cpu = torch.randn(4, 4)
        x = x_cpu.to(self.device)
        x.cos_()
        self.assertEqual(x.cpu(), x_cpu.cos())

    # --- Parameterized unary in-place ---

    def test_gelu_inplace(self):
        x_cpu = torch.randn(4, 4)
        x = x_cpu.to(self.device)
        x_ref = F.gelu(x_cpu)
        torch.ops.aten.gelu_(x)
        self.assertEqual(x.cpu(), x_ref, atol=1e-5, rtol=1e-5)

    def test_hardtanh_inplace(self):
        x_cpu = torch.randn(4, 4) * 3
        x = x_cpu.to(self.device)
        F.hardtanh(x, min_val=-2.0, max_val=2.0, inplace=True)
        self.assertEqual(x.cpu(), x_cpu.clamp(-2.0, 2.0))

    def test_leaky_relu_inplace(self):
        x_cpu = torch.randn(4, 4)
        x = x_cpu.to(self.device)
        x_ref = F.leaky_relu(x_cpu, negative_slope=0.2)
        F.leaky_relu(x, negative_slope=0.2, inplace=True)
        self.assertEqual(x.cpu(), x_ref)

    def test_elu_inplace(self):
        x_cpu = torch.randn(4, 4)
        x = x_cpu.to(self.device)
        x_ref = F.elu(x_cpu, alpha=1.5)
        F.elu(x, alpha=1.5, inplace=True)
        self.assertEqual(x.cpu(), x_ref, atol=1e-5, rtol=1e-5)

    # --- Scalar binary in-place ---

    def test_add_scalar_inplace(self):
        x_cpu = torch.randn(4, 4)
        x = x_cpu.to(self.device)
        x.add_(3.0)
        self.assertEqual(x.cpu(), x_cpu + 3.0)

    def test_mul_scalar_inplace(self):
        x_cpu = torch.randn(4, 4)
        x = x_cpu.to(self.device)
        x.mul_(0.5)
        self.assertEqual(x.cpu(), x_cpu * 0.5)

    def test_div_scalar_inplace(self):
        x_cpu = torch.randn(4, 4)
        x = x_cpu.to(self.device)
        x.div_(2.0)
        self.assertEqual(x.cpu(), x_cpu / 2.0)

    def test_sub_scalar_inplace(self):
        x_cpu = torch.randn(4, 4)
        x = x_cpu.to(self.device)
        x.sub_(1.5)
        self.assertEqual(x.cpu(), x_cpu - 1.5)

    def test_pow_scalar_inplace(self):
        x_cpu = torch.rand(4, 4) + 0.5
        x = x_cpu.to(self.device)
        x.pow_(2.0)
        self.assertEqual(x.cpu(), x_cpu.pow(2.0), atol=1e-5, rtol=1e-5)

    # --- Non-contiguous in-place (temp + strided copy) ---

    def test_noncontiguous_relu_inplace(self):
        x_cpu = torch.randn(4, 4)
        x = x_cpu.to(self.device)
        xt = x.t()
        xt.relu_()
        self.assertEqual(xt.cpu(), x_cpu.t().relu())

    def test_noncontiguous_add_inplace(self):
        x_cpu = torch.randn(4, 4)
        y_cpu = torch.randn(4, 4)
        x = x_cpu.to(self.device)
        y = y_cpu.to(self.device)
        xt = x.t()
        yt = y.t()
        xt.add_(yt)
        self.assertEqual(xt.cpu(), x_cpu.t() + y_cpu.t())

    # --- += / -= / *= operators ---

    def test_iadd_operator(self):
        x_cpu = torch.randn(4, 4)
        y_cpu = torch.randn(4, 4)
        x = x_cpu.to(self.device)
        y = y_cpu.to(self.device)
        x += y
        self.assertEqual(x.cpu(), x_cpu + y_cpu)

    def test_imul_operator(self):
        x_cpu = torch.randn(4, 4)
        x = x_cpu.to(self.device)
        x *= 2.0
        self.assertEqual(x.cpu(), x_cpu * 2.0)


if __name__ == "__main__":
    run_tests()
