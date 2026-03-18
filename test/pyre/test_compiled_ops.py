"""Tests for compiled kernel dispatch (Epic 1).

These tests require the IREE compiler to be available. Set PYRE_IREE_COMPILE
or PYRE_IREE_COMPILER_LIB environment variable before running.
"""
import os
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


def has_iree_compiler():
    """Check if IREE compiler is available via env vars."""
    return bool(
        os.environ.get("PYRE_IREE_COMPILE")
        or os.environ.get("PYRE_IREE_COMPILER_LIB")
    )


@unittest.skipUnless(has_iree_compiler(), "IREE compiler not available")
class TestCompiledAdd(TestCase):
    """Test aten::add through compiled kernel dispatch."""

    def _host_device(self):
        return torch.device("host:0")

    def test_add_basic_f32(self):
        a = torch.randn(4, 4, device=self._host_device())
        b = torch.randn(4, 4, device=self._host_device())
        result = torch.add(a, b)
        expected = a.cpu() + b.cpu()
        self.assertEqual(result.cpu(), expected)

    def test_add_alpha(self):
        a = torch.randn(4, 4, device=self._host_device())
        b = torch.randn(4, 4, device=self._host_device())
        result = torch.add(a, b, alpha=2.0)
        expected = a.cpu() + 2.0 * b.cpu()
        self.assertEqual(result.cpu(), expected)

    def test_add_broadcast(self):
        a = torch.randn(4, 4, device=self._host_device())
        b = torch.randn(1, 4, device=self._host_device())
        result = torch.add(a, b)
        expected = a.cpu() + b.cpu()
        self.assertEqual(result.cpu(), expected)

    def test_add_int32(self):
        a = torch.randint(0, 100, (4, 4), dtype=torch.int32,
                          device=self._host_device())
        b = torch.randint(0, 100, (4, 4), dtype=torch.int32,
                          device=self._host_device())
        result = torch.add(a, b)
        expected = a.cpu() + b.cpu()
        self.assertEqual(result.cpu(), expected)


@unittest.skipUnless(has_iree_compiler(), "IREE compiler not available")
class TestCompiledUnary(TestCase):
    """Test unary ops through compiled kernel dispatch."""

    def _host_device(self):
        return torch.device("host:0")

    def test_neg_f32(self):
        a = torch.randn(4, 4, device=self._host_device())
        result = torch.neg(a)
        expected = -a.cpu()
        self.assertEqual(result.cpu(), expected)

    def test_abs_f32(self):
        a = torch.randn(4, 4, device=self._host_device())
        result = torch.abs(a)
        expected = torch.abs(a.cpu())
        self.assertEqual(result.cpu(), expected)

    def test_relu_f32(self):
        a = torch.randn(4, 4, device=self._host_device())
        result = torch.relu(a)
        expected = torch.relu(a.cpu())
        self.assertEqual(result.cpu(), expected)


@unittest.skipUnless(has_iree_compiler(), "IREE compiler not available")
class TestCompiledBinary(TestCase):
    """Test binary ops through compiled kernel dispatch."""

    def _host_device(self):
        return torch.device("host:0")

    def test_sub_f32(self):
        a = torch.randn(4, 4, device=self._host_device())
        b = torch.randn(4, 4, device=self._host_device())
        result = torch.sub(a, b)
        expected = a.cpu() - b.cpu()
        self.assertEqual(result.cpu(), expected)

    def test_mul_f32(self):
        a = torch.randn(4, 4, device=self._host_device())
        b = torch.randn(4, 4, device=self._host_device())
        result = torch.mul(a, b)
        expected = a.cpu() * b.cpu()
        self.assertEqual(result.cpu(), expected)

    def test_div_f32(self):
        a = torch.randn(4, 4, device=self._host_device())
        b = torch.randn(4, 4, device=self._host_device()) + 0.1  # avoid div by 0
        result = torch.div(a, b)
        expected = a.cpu() / b.cpu()
        self.assertEqual(result.cpu(), expected, atol=1e-5, rtol=1e-5)


@unittest.skipUnless(has_iree_compiler(), "IREE compiler not available")
class TestAsyncPipeline(TestCase):
    """Test that chained operations execute correctly (fences chain)."""

    def _host_device(self):
        return torch.device("host:0")

    def test_chained_ops(self):
        """a + b → relu → * c should produce correct results."""
        a = torch.randn(4, 4, device=self._host_device())
        b = torch.randn(4, 4, device=self._host_device())
        c = torch.randn(4, 4, device=self._host_device())
        result = torch.mul(torch.relu(torch.add(a, b)), c)
        expected = torch.mul(torch.relu(a.cpu() + b.cpu()), c.cpu())
        self.assertEqual(result.cpu(), expected)


if __name__ == "__main__":
    run_tests()
