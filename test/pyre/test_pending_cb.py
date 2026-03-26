"""Tests for pending command buffer infrastructure (Epic 7, T1).

Validates that the getOrCreateCB / flush / flushIfEager plumbing on
PyreStream doesn't regress existing operations. The pending CB
infrastructure is exercised implicitly through all native dispatches
(copy, fill) once T5 migrates them. For now, these tests validate that
the new methods exist and that the synchronize path (which now calls
flush()) doesn't break anything.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestPendingCB(TestCase):
    def test_synchronize_with_no_pending(self):
        """synchronize() is a no-op when no pending CB exists."""
        t = torch.ones(4, device="host:0")
        result = t.to("cpu")
        self.assertEqual(result, torch.ones(4))

    def test_fill_copy_add_sequence(self):
        """fill → copy → add must produce correct results.

        This exercises the flush ordering between native ops (fill, copy)
        and envelope dispatches (add). Once T5 migrates native ops to the
        pending CB, this test validates that flush before envelope invoke
        keeps ordering correct.
        """
        t = torch.zeros(8, device="host:0")
        t.fill_(3.0)
        t2 = t.clone()
        result = (t2 + 1.0).to("cpu")
        self.assertEqual(result, torch.full((8,), 4.0))

    def test_multiple_fills_before_compute(self):
        """Multiple fills before a compute op should all complete."""
        a = torch.empty(4, device="host:0")
        b = torch.empty(4, device="host:0")
        a.fill_(2.0)
        b.fill_(3.0)
        result = (a + b).to("cpu")
        self.assertEqual(result, torch.full((4,), 5.0))

    def test_interleaved_native_and_compute(self):
        """Interleaving native (copy) and compute (add, mul) ops."""
        x = torch.randn(16, device="host:0")
        y = x.clone()      # copy (native)
        z = y + 1.0        # add (envelope)
        w = z.clone()       # copy (native)
        r = w * 2.0         # mul (envelope)
        expected = (x.cpu() + 1.0) * 2.0
        self.assertEqual(r.to("cpu"), expected)

    def test_synchronize_after_operations(self):
        """synchronize should flush and wait for all pending work."""
        t = torch.randn(32, device="host:0")
        result = t + t
        # synchronize (implicit in .to("cpu"))
        self.assertEqual(result.to("cpu"), t.to("cpu") * 2)

    def test_fill_then_envelope_ordering(self):
        """fill (pending CB) → add (envelope) must flush before VM invoke.

        This is the key T6 test: fill records into the pending CB, then
        add triggers flush() + VM invoke. If flush is missing, the add
        sees uninitialized memory.
        """
        t = torch.zeros(16, device="host:0")
        t.fill_(1.0)
        result = (t + 1.0).to("cpu")
        self.assertEqual(result, torch.full((16,), 2.0))

    def test_copy_then_envelope_ordering(self):
        """copy (pending CB) → mul (envelope) must flush before VM invoke."""
        src = torch.randn(8, device="host:0")
        dst = src.clone()  # copy via pending CB
        result = (dst * 2.0).to("cpu")
        expected = src.to("cpu") * 2.0
        self.assertEqual(result, expected)

    def test_large_matmul_transient(self):
        """1024x1024 matmul exercises the transient size query (was 1MiB crash)."""
        a = torch.randn(1024, 1024, device="host:0")
        b = torch.randn(1024, 1024, device="host:0")
        c = a @ b
        self.assertEqual(c.shape, (1024, 1024))

    def test_rapid_alloc_free_with_sync(self):
        """Rapid alloc/free cycles should not desync the timeline."""
        for _ in range(100):
            t = torch.randn(64, device="host:0")
            _ = t + 1.0
        # Final op must still work
        t = torch.ones(4, device="host:0")
        self.assertEqual((t + t).to("cpu"), torch.full((4,), 2.0))


if __name__ == "__main__":
    run_tests()
