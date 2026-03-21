"""Test mm with transposed (non-contiguous) inputs."""
import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestMmTransposed(TestCase):
    def test_mm_contiguous(self):
        a = torch.randn(2, 3).to("host:0")
        b = torch.randn(3, 4).to("host:0")
        r = torch.mm(a, b)
        ref = torch.mm(a.cpu(), b.cpu())
        self.assertEqual(r.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_mm_rhs_transposed(self):
        """F.linear does mm(x, w.T) — w.T is non-contiguous."""
        a = torch.randn(2, 3).to("host:0")
        w = torch.randn(4, 3).to("host:0")  # [4,3]
        wt = w.t()  # [3,4] non-contiguous
        self.assertFalse(wt.is_contiguous())
        r = torch.mm(a, wt)
        ref = torch.mm(a.cpu(), w.cpu().t())
        self.assertEqual(r.shape, torch.Size([2, 4]))
        self.assertEqual(r.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_mm_lhs_transposed(self):
        a = torch.randn(3, 2).to("host:0")
        b = torch.randn(3, 4).to("host:0")
        r = torch.mm(a.t(), b)  # [2,3] @ [3,4] = [2,4]
        ref = torch.mm(a.cpu().t(), b.cpu())
        self.assertEqual(r.shape, torch.Size([2, 4]))
        self.assertEqual(r.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_addmm_rhs_transposed(self):
        """addmm with transposed weight (common in F.linear with bias)."""
        bias = torch.randn(4).to("host:0")
        a = torch.randn(2, 3).to("host:0")
        w = torch.randn(4, 3).to("host:0")
        wt = w.t()  # [3,4] non-contiguous
        r = torch.addmm(bias, a, wt)
        ref = torch.addmm(bias.cpu(), a.cpu(), w.cpu().t())
        self.assertEqual(r.shape, torch.Size([2, 4]))
        self.assertEqual(r.cpu(), ref, atol=1e-4, rtol=1e-4)

    def test_linear(self):
        """F.linear = mm(input, weight.T) + bias."""
        x = torch.randn(1, 1, 64).to("host:0")
        w = torch.randn(192, 64).to("host:0")
        r = torch.nn.functional.linear(x, w)
        ref = torch.nn.functional.linear(x.cpu(), w.cpu())
        self.assertEqual(r.shape, torch.Size([1, 1, 192]))
        self.assertEqual(r.cpu(), ref, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    run_tests()
