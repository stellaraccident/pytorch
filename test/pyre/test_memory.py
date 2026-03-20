"""Tests for pyre storage, allocator, and buffer operations."""

import gc
import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestStorageAllocator(TestCase):
    def test_basic_allocation(self):
        for size in [1, 4, 16, 256, 1024, 65536]:
            t = torch.empty(size, device="host:0")
            self.assertEqual(t.numel(), size)

    def test_fill_scalar_float32(self):
        t = torch.ones(4, device="host:0")
        result = t.to("cpu")
        self.assertEqual(result, torch.ones(4))

    def test_fill_scalar_int32(self):
        t = torch.ones(4, device="host:0", dtype=torch.int32)
        result = t.to("cpu")
        self.assertEqual(result, torch.ones(4, dtype=torch.int32))

    def test_fill_supported_dtypes(self):
        for dtype in [torch.int8, torch.int16, torch.int32,
                      torch.float32, torch.float16, torch.bfloat16]:
            t = torch.ones(4, device="host:0", dtype=dtype)
            self.assertEqual(t.to("cpu").sum().item(), 4.0,
                             msg=f"fill failed for {dtype}")

    def test_fill_8byte_types(self):
        for dtype in [torch.float64, torch.int64]:
            t = torch.ones(4, device="host:0", dtype=dtype)
            self.assertEqual(t.to("cpu").sum().item(), 4.0,
                             msg=f"8-byte fill failed for {dtype}")

    def test_fill_zero_value(self):
        t = torch.zeros(8, device="host:0")
        result = t.to("cpu")
        self.assertEqual(result, torch.zeros(8))

    def test_fill_nonzero_value(self):
        t = torch.full((4,), 42.0, device="host:0")
        result = t.to("cpu")
        self.assertEqual(result, torch.full((4,), 42.0))

    def test_storage_device(self):
        t = torch.ones(4, device="host:0")
        self.assertEqual(t.storage().device.type, "host")


class TestCopyOperations(TestCase):
    def test_cpu_to_host(self):
        t_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0])
        t_host = t_cpu.to("host:0")
        self.assertEqual(t_host.device.type, "host")

    def test_host_to_cpu(self):
        t_host = torch.ones(4, device="host:0")
        t_cpu = t_host.to("cpu")
        self.assertEqual(t_cpu, torch.ones(4))

    def test_roundtrip_preserves_values(self):
        t_orig = torch.arange(10, dtype=torch.float32)
        t_back = t_orig.to("host:0").to("cpu")
        self.assertEqual(t_orig, t_back)

    def test_host_to_host_copy(self):
        a = torch.ones(4, device="host:0")
        b = torch.empty(4, device="host:0")
        b.copy_(a)
        self.assertEqual(b.to("cpu"), torch.ones(4))

    def test_copy_with_dtype_conversion(self):
        # CPU int → host float
        t_int = torch.tensor([1, 2, 3], dtype=torch.int32)
        t_host = t_int.to(device="host:0", dtype=torch.float32)
        t_back = t_host.to("cpu")
        self.assertEqual(t_back, torch.tensor([1.0, 2.0, 3.0]))

    def test_large_transfer(self):
        # > 64KB to exercise the staging buffer path
        size = 100000  # 400KB as float32
        t_cpu = torch.arange(size, dtype=torch.float32)
        t_host = t_cpu.to("host:0")
        t_back = t_host.to("cpu")
        self.assertEqual(t_cpu, t_back)

    def test_local_scalar_dense(self):
        t = torch.ones(1, device="host:0")
        self.assertEqual(t.item(), 1.0)

    def test_local_scalar_dense_int(self):
        t = torch.full((1,), 42, device="host:0", dtype=torch.int32)
        self.assertEqual(t.item(), 42)


class TestDestructionOrder(TestCase):
    """Verify that buffer teardown completes without crashes across
    various allocation/deallocation patterns. These exercise the
    PyreBufferContext destructor's use-barrier wait and the device
    teardown sequencing."""

    def test_rapid_alloc_dealloc(self):
        for _ in range(200):
            t = torch.ones(1024, device="host:0")
            del t
        gc.collect()

    def test_interleaved_transfers(self):
        for _ in range(100):
            t_cpu = torch.randn(256)
            t_host = t_cpu.to("host:0")
            t_back = t_host.to("cpu")
            del t_cpu, t_host, t_back
        gc.collect()

    def test_varying_sizes(self):
        for i in range(100):
            size = (i + 1) * 100
            t = torch.empty(size, device="host:0")
            del t
        gc.collect()

    def test_multiple_live_tensors(self):
        tensors = [torch.ones(256, device="host:0") for _ in range(50)]
        del tensors
        gc.collect()


class TestNonContiguousCopy(TestCase):
    def test_d2d_copy_transposed(self):
        x = torch.randn(4, 8, device="host:0")
        y = x.t()  # non-contiguous
        z = y.clone()  # triggers d2d copy via CPU roundtrip
        self.assertEqual(z.cpu(), y.cpu())

    def test_d2d_copy_sliced(self):
        x = torch.randn(8, 8, device="host:0")
        y = x[::2]  # non-contiguous
        z = y.clone()
        self.assertEqual(z.cpu(), y.cpu())

    def test_fill_noncontiguous(self):
        x = torch.randn(4, 8, device="host:0")
        y = x[:, ::2]
        y.fill_(0.0)
        self.assertTrue((y.cpu() == 0).all())

    def test_fill_noncontiguous_nonzero(self):
        x = torch.randn(4, 8, device="host:0")
        y = x[:, ::2]
        y.fill_(42.0)
        self.assertTrue((y.cpu() == 42.0).all())

    def test_fill_float64(self):
        t = torch.zeros(4, dtype=torch.float64, device="host:0")
        t.fill_(3.14)
        self.assertTrue(torch.allclose(t.cpu(),
                        torch.full((4,), 3.14, dtype=torch.float64)))

    def test_d2d_copy_narrowed(self):
        x = torch.randn(4, 8, device="host:0")
        y = x[:, :3]  # narrowed columns
        z = y.clone()
        self.assertEqual(z.cpu(), y.cpu())

    def test_d2d_copy_permuted_3d(self):
        x = torch.randn(2, 3, 4, device="host:0")
        y = x.permute(2, 0, 1)  # non-contiguous
        z = y.clone()
        self.assertEqual(z.cpu(), y.cpu())

    def test_d2d_copy_offset_transpose(self):
        # Storage offset + transpose → Tier 2 with base offsets.
        x = torch.randn(8, 8, device="host:0")
        y = x[2:].t()
        z = y.clone()
        self.assertEqual(z.cpu(), y.cpu())

    def test_cpu_to_noncontiguous_dst(self):
        # CPU→device where dst is non-contiguous: should work via temp+copy.
        src = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        dst = torch.empty(4, 8, device="host:0")[:, ::2]
        dst.copy_(src.to("host:0"))
        self.assertEqual(dst.cpu(), src)


if __name__ == "__main__":
    run_tests()
