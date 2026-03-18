"""Tests for pyre host device management."""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestDevice(TestCase):
    def test_device_count(self):
        # Phase 0: single host device
        self.assertGreaterEqual(torch.accelerator.device_count(), 1)

    def test_device_type(self):
        t = torch.empty(1, device="host:0")
        self.assertEqual(t.device.type, "host")
        self.assertEqual(t.device.index, 0)

    def test_tensor_device_assignment(self):
        t = torch.empty(4, device="host:0")
        self.assertEqual(t.device, torch.device("host", 0))

    def test_device_guard(self):
        with torch.device("host:0"):
            t = torch.empty(4)
            self.assertEqual(t.device.type, "host")
            self.assertEqual(t.device.index, 0)


class TestDeviceCreation(TestCase):
    def test_empty(self):
        t = torch.empty(4, device="host:0")
        self.assertEqual(t.shape, (4,))
        self.assertEqual(t.device.type, "host")

    def test_empty_strided(self):
        t = torch.empty_strided((2, 3), (3, 1), device="host:0")
        self.assertEqual(t.shape, (2, 3))
        self.assertEqual(t.stride(), (3, 1))
        self.assertTrue(t.is_contiguous())

    def test_empty_different_dtypes(self):
        for dtype in [torch.float32, torch.float16, torch.bfloat16,
                      torch.int8, torch.int16, torch.int32]:
            t = torch.empty(4, device="host:0", dtype=dtype)
            self.assertEqual(t.dtype, dtype)
            self.assertEqual(t.device.type, "host")

    def test_empty_zero_size(self):
        t = torch.empty(0, device="host:0")
        self.assertEqual(t.numel(), 0)

    def test_tensor_repr(self):
        t = torch.ones(4, device="host:0")
        s = repr(t)
        self.assertIn("host:0", s)
        self.assertIn("1.", s)

    def test_tensor_repr_2d(self):
        t = torch.ones(2, 3, device="host:0")
        s = repr(t)
        self.assertIn("host:0", s)


if __name__ == "__main__":
    run_tests()
