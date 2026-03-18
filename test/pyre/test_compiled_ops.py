"""Tests for compiled kernel dispatch (Epic 1).

Requires IREE compiler: set PYRE_IREE_COMPILE or PYRE_IREE_COMPILER_LIB.
"""
import os
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


def has_compiler():
    return bool(
        os.environ.get("PYRE_IREE_COMPILE")
        or os.environ.get("PYRE_IREE_COMPILER_LIB")
    )


DEVICE = "host:0"


@unittest.skipUnless(has_compiler(), "IREE compiler not available")
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


@unittest.skipUnless(has_compiler(), "IREE compiler not available")
class TestUnaryOps(TestCase):
    """Unary elementwise ops: neg, relu."""

    def test_neg_f32(self):
        a = torch.randn(4, 4, device=DEVICE)
        self.assertEqual(torch.neg(a).cpu(), -a.cpu())

    def test_neg_f32_1d(self):
        a = torch.randn(16, device=DEVICE)
        self.assertEqual(torch.neg(a).cpu(), -a.cpu())

    def test_relu_f32(self):
        a = torch.randn(4, 4, device=DEVICE)
        self.assertEqual(torch.relu(a).cpu(), torch.relu(a.cpu()))

    def test_relu_mixed_sign(self):
        a = torch.tensor([-1.0, 0.0, 1.0, -0.5, 2.0], device=DEVICE)
        expected = torch.tensor([0.0, 0.0, 1.0, 0.0, 2.0])
        self.assertEqual(torch.relu(a).cpu(), expected)


@unittest.skipUnless(has_compiler(), "IREE compiler not available")
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


@unittest.skipUnless(has_compiler(), "IREE compiler not available")
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


if __name__ == "__main__":
    run_tests()
