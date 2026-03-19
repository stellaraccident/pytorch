"""Integration test: MLP forward pass on host:0.

Linear layers use CPU fallback (matmul not yet compiled).
Elementwise ops (add, relu) use compiled IREE kernels when available.
"""
import os
import unittest

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase


def has_compiler():
    return bool(
        os.environ.get("PYRE_IREE_COMPILE")
        or os.environ.get("PYRE_IREE_COMPILER_LIB")
    )


DEVICE = "host:0"


class TestMLP(TestCase):
    """MLP forward pass: the Epic 1 success metric."""

    def _make_mlp(self, in_features, hidden, out_features):
        model = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )
        return model

    @unittest.skipUnless(has_compiler(), "IREE compiler not available")
    def test_mlp_forward_cpu_reference(self):
        """MLP on host:0 matches CPU reference."""
        torch.manual_seed(42)
        model_cpu = self._make_mlp(16, 32, 8)
        # Make all parameters contiguous before moving to device
        # (weight matrices from nn.Linear may be transposed).
        model_host = self._make_mlp(16, 32, 8)
        model_host.load_state_dict(model_cpu.state_dict())
        model_host = model_host.to(DEVICE)

        x_cpu = torch.randn(4, 16)
        x_host = x_cpu.to(DEVICE)

        with torch.no_grad():
            y_cpu = model_cpu(x_cpu)
            y_host = model_host(x_host)

        self.assertEqual(y_host.cpu(), y_cpu, atol=1e-5, rtol=1e-5)

    @unittest.skipUnless(has_compiler(), "IREE compiler not available")
    def test_mlp_different_batch_sizes(self):
        """MLP works with varying batch sizes."""
        torch.manual_seed(0)
        model = self._make_mlp(8, 16, 4).to(DEVICE)

        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 8, device=DEVICE)
            with torch.no_grad():
                y = model(x)
            self.assertEqual(y.shape, (batch_size, 4))
            # Verify not all zeros (computation actually happened)
            self.assertGreater(y.cpu().abs().sum().item(), 0)

    @unittest.skipUnless(has_compiler(), "IREE compiler not available")
    def test_mlp_with_manual_elementwise(self):
        """Manually compose linear + relu using ops that go through compiled path."""
        torch.manual_seed(42)
        W1 = torch.randn(8, 4, device=DEVICE)
        b1 = torch.randn(4, device=DEVICE)
        W2 = torch.randn(4, 4, device=DEVICE)
        b2 = torch.randn(4, device=DEVICE)
        x = torch.randn(2, 8, device=DEVICE)

        # Manual forward: mm via CPU fallback, add + relu via compiled
        h = torch.mm(x, W1)  # CPU fallback
        h = torch.add(h, b1)  # compiled (if buffers ok)
        h = torch.relu(h)  # compiled
        y = torch.mm(h, W2)  # CPU fallback
        y = torch.add(y, b2)  # compiled

        # CPU reference
        h_cpu = torch.mm(x.cpu(), W1.cpu()) + b1.cpu()
        h_cpu = torch.relu(h_cpu)
        y_cpu = torch.mm(h_cpu, W2.cpu()) + b2.cpu()

        self.assertEqual(y.cpu(), y_cpu, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    run_tests()
