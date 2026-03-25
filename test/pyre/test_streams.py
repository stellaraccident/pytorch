"""Tests for pyre stream and timeline management."""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestStream(TestCase):
    def test_default_stream(self):
        s = torch.Stream(device="host:0")
        self.assertEqual(s.device_index, 0)

    def test_stream_from_pool(self):
        # getNewStream should return a stream from the pool
        s1 = torch.Stream(device="host:0")
        s2 = torch.Stream(device="host:0")
        # Both are valid streams on the same device
        self.assertEqual(s1.device_index, 0)
        self.assertEqual(s2.device_index, 0)

    def test_operations_complete(self):
        # Operations on the default stream complete correctly
        t = torch.ones(4, device="host:0")
        result = t.to("cpu")
        self.assertEqual(result, torch.ones(4))

    def test_operations_on_default_stream(self):
        # Basic operations should work on the default stream
        t = torch.ones(4, device="host:0")
        t2 = torch.empty(4, device="host:0")
        t2.copy_(t)
        result = t2.to("cpu")
        self.assertEqual(result, torch.ones(4))


class TestErrorRecovery(TestCase):
    def test_dispatch_error_does_not_hang(self):
        """A failed dispatch must not leave the timeline desynchronized."""
        x = torch.randn(4, 4, device="host:0")
        y = torch.randn(4, 4, device="host:0")

        # Force a dispatch failure via incompatible types for a compiled op.
        try:
            # bool tensors aren't supported by most compiled ops
            bad = torch.ones(4, 4, dtype=torch.bool, device="host:0")
            _ = bad + x
        except RuntimeError:
            pass

        # This MUST NOT hang — the timeline should be consistent.
        z = x + y
        expected = x.cpu() + y.cpu()
        self.assertEqual(z.cpu(), expected)

    def test_multiple_errors_recover(self):
        """Multiple consecutive failures should all recover."""
        x = torch.randn(4, device="host:0")
        for _ in range(3):
            try:
                bad = torch.ones(4, dtype=torch.bool, device="host:0")
                _ = bad + x
            except RuntimeError:
                pass
        # Still works after repeated failures.
        result = (x + x).cpu()
        self.assertEqual(result, x.cpu() * 2)


if __name__ == "__main__":
    run_tests()
