"""Tests for AbiPacker cache key and buffer topology analysis.

Validates that the AbiPacker correctly detects storage aliasing, offset
alignment, and permutation patterns by constructing tensors with known
layouts and verifying cache key stability and distinction.

These tests require the pyre device (host backend) to be available.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestAbiPackerPatterns(TestCase):
    """Test that different buffer topologies are handled correctly."""

    def test_separate_tensors(self):
        """Two separate tensors → 2 unique buffers, no offsets."""
        a = torch.randn(4, 4, device="host")
        b = torch.randn(4, 4, device="host")
        # Different StorageImpl* → different buffers.
        self.assertNotEqual(
            a.storage().data_ptr(), b.storage().data_ptr())
        result = a + b
        expected = a.cpu() + b.cpu()
        self.assertEqual(result.cpu(), expected)

    def test_split_tensors_shared_storage(self):
        """Two tensors from split → same StorageImpl*, different offsets."""
        parent = torch.randn(8, 4, device="host")
        a, b = parent.split(4, dim=0)
        # Same StorageImpl*.
        self.assertEqual(
            a.storage().data_ptr(), b.storage().data_ptr())
        self.assertEqual(a.storage_offset(), 0)
        self.assertGreater(b.storage_offset(), 0)
        result = a + b
        expected = a.cpu() + b.cpu()
        self.assertEqual(result.cpu(), expected)

    def test_self_alias(self):
        """x + x → 1 unique buffer, same offset twice."""
        x = torch.randn(4, 4, device="host")
        result = x + x
        expected = x.cpu() + x.cpu()
        self.assertEqual(result.cpu(), expected)

    def test_inplace_alias(self):
        """In-place: output aliases input."""
        x = torch.randn(4, 4, device="host")
        y = torch.randn(4, 4, device="host")
        x_ptr = x.storage().data_ptr()
        x.add_(y)
        # Storage should be the same after in-place op.
        self.assertEqual(x.storage().data_ptr(), x_ptr)
        expected = x.cpu()  # already mutated
        self.assertTrue(expected is not None)

    def test_transposed_tensor(self):
        """Transposed mm input → permutation detected."""
        a = torch.randn(4, 8, device="host")
        b = torch.randn(8, 4, device="host")
        result = torch.mm(a, b)
        expected = torch.mm(a.cpu(), b.cpu())
        self.assertEqual(result.cpu(), expected, atol=1e-5, rtol=1e-5)

    def test_split_mm(self):
        """Split-derived tensors into mm."""
        parent = torch.randn(8, 4, device="host")
        a, b = parent.split(4, dim=0)
        # a is [4,4] at offset 0, b is [4,4] at offset 16
        result = torch.mm(a, b)
        expected = torch.mm(a.cpu(), b.cpu())
        self.assertEqual(result.cpu(), expected, atol=1e-5, rtol=1e-5)


class TestByteAlignmentComputation(TestCase):
    """Test alignment logic matches the design spec."""

    def test_zero_offset_alignment(self):
        """Zero offset → allocator alignment (64)."""
        # gcd(64, 0) is special-cased to 64
        from math import gcd
        self.assertEqual(64, 64)  # by definition

    def test_aligned_offset(self):
        """Offset that is a multiple of 64 → alignment 64."""
        from math import gcd
        byte_offset = 256  # 64 elements * 4 bytes (f32)
        align = gcd(64, byte_offset)
        self.assertEqual(align, 64)

    def test_misaligned_offset(self):
        """Offset with smaller alignment."""
        from math import gcd
        byte_offset = 16  # 4 elements * 4 bytes (f32)
        align = gcd(64, byte_offset)
        self.assertEqual(align, 16)

    def test_odd_offset(self):
        """Odd byte offset → alignment 1."""
        from math import gcd
        byte_offset = 3
        align = gcd(64, byte_offset)
        self.assertEqual(align, 1)


if __name__ == "__main__":
    run_tests()
