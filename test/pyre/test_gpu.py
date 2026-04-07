"""Tests for the Pyre GPU frontend over DeviceType::HIP."""

from functools import lru_cache
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


@lru_cache(maxsize=None)
def pyre_gpu_available():
    try:
        tensor = torch.empty(1, device="hip:0")
    except RuntimeError:
        return False
    return tensor.device.type == "gpu"


@lru_cache(maxsize=None)
def pyre_gpu_compiler_available():
    if not pyre_gpu_available():
        return False
    try:
        lhs = torch.randn(4, device="hip:0")
        rhs = torch.randn(4, device="hip:0")
        _ = (lhs + rhs).cpu()
    except RuntimeError:
        return False
    return True


@unittest.skipUnless(pyre_gpu_available(), "pyre gpu backend is unavailable")
class TestGpuDevice(TestCase):
    def test_gpu_and_hip_device_aliases(self):
        gpu_tensor = torch.empty(4, device="gpu:0")
        hip_tensor = torch.empty(4, device="hip:0")
        self.assertEqual(gpu_tensor.device, torch.device("gpu", 0))
        self.assertEqual(hip_tensor.device, torch.device("gpu", 0))
        self.assertEqual(gpu_tensor.device.type, "gpu")
        self.assertEqual(hip_tensor.device.type, "gpu")

    def test_empty(self):
        tensor = torch.empty(8, device="gpu:0")
        self.assertEqual(tensor.shape, (8,))
        self.assertEqual(tensor.device.type, "gpu")
        self.assertEqual(tensor.device.index, 0)


@unittest.skipUnless(pyre_gpu_available(), "pyre gpu backend is unavailable")
class TestGpuTransfer(TestCase):
    def test_cpu_to_gpu_to_cpu(self):
        src = torch.arange(8, dtype=torch.float32)
        gpu_tensor = src.to("gpu:0")
        self.assertEqual(gpu_tensor.device, torch.device("gpu", 0))
        self.assertEqual(gpu_tensor.cpu(), src)

    def test_gpu_to_gpu_copy(self):
        src = torch.arange(8, dtype=torch.float32).to("gpu:0")
        dst = torch.empty(8, device="gpu:0")
        dst.copy_(src)
        self.assertEqual(dst.cpu(), src.cpu())


@unittest.skipUnless(pyre_gpu_available(), "pyre gpu backend is unavailable")
class TestGpuStream(TestCase):
    def test_stream_ordering(self):
        stream = torch.Stream(device="gpu:0")
        with stream:
            src = torch.arange(8, dtype=torch.float32).to("gpu:0")
            dst = torch.empty(8, device="gpu:0")
            dst.copy_(src)
        self.assertEqual(dst.cpu(), src.cpu())


@unittest.skipUnless(
    pyre_gpu_compiler_available(),
    "pyre gpu compiler backend is unavailable",
)
class TestGpuCompiledOps(TestCase):
    def test_add(self):
        lhs = torch.randn(4, 4, device="gpu:0")
        rhs = torch.randn(4, 4, device="gpu:0")
        out = lhs + rhs
        self.assertEqual(out.cpu(), lhs.cpu() + rhs.cpu(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    run_tests()
