"""Tests for AbiGenerator envelope MLIR generation.

Validates that generated envelope MLIR compiles with iree-compile.
These test the MLIR patterns directly (matching the validated prototypes)
before the full AbiGenerator is wired into dispatch (T4).
"""

import os
import subprocess
import tempfile
import unittest

from torch.testing._internal.common_utils import run_tests, TestCase

IREE_COMPILE = os.environ.get("PYRE_IREE_COMPILE", "")


def iree_compile_available():
    return IREE_COMPILE and os.path.isfile(IREE_COMPILE)


def compile_mlir(mlir_text, input_type="torch"):
    """Compile MLIR text with iree-compile. Returns (success, stderr)."""
    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as f:
        f.write(mlir_text)
        f.flush()
        mlir_path = f.name

    out_path = mlir_path + ".vmfb"
    try:
        result = subprocess.run(
            [IREE_COMPILE,
             f"--iree-input-type={input_type}",
             "--iree-hal-target-backends=llvm-cpu",
             "--iree-llvmcpu-target-cpu=host",
             mlir_path, "-o", out_path],
            capture_output=True, text=True, timeout=60)
        return result.returncode == 0, result.stderr
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        for p in [mlir_path, out_path]:
            if os.path.exists(p):
                os.unlink(p)


class TestEnvelopeCompilation(TestCase):
    """Verify generated envelope patterns compile with iree-compile."""

    def setUp(self):
        if not iree_compile_available():
            self.skipTest("PYRE_IREE_COMPILE not set or not found")

    def test_v1_simple_add_envelope(self):
        """V1: Separate buffers, no offset, torch compute."""
        mlir = """\
module @pyre_kernel {
  func.func private @compute(
      %a_builtin: tensor<?x4xf32>, %b_builtin: tensor<?x4xf32>
  ) -> tensor<?x4xf32>
      attributes {inlining_policy = #util.inline.always} {
    %a = torch_c.from_builtin_tensor %a_builtin
        : tensor<?x4xf32> -> !torch.vtensor<[?,4], f32>
    %b = torch_c.from_builtin_tensor %b_builtin
        : tensor<?x4xf32> -> !torch.vtensor<[?,4], f32>
    %alpha = torch.constant.int 1
    %result = torch.aten.add.Tensor %a, %b, %alpha
        : !torch.vtensor<[?,4], f32>, !torch.vtensor<[?,4], f32>, !torch.int
        -> !torch.vtensor<[?,4], f32>
    %result_builtin = torch_c.to_builtin_tensor %result
        : !torch.vtensor<[?,4], f32> -> tensor<?x4xf32>
    return %result_builtin : tensor<?x4xf32>
  }

  util.func public @add_envelope(
      %buf0: !hal.buffer_view,
      %buf1: !hal.buffer_view,
      %dim0: index,
      %buf_out: !hal.buffer,
      %transients: !hal.buffer,
      %wait: !hal.fence,
      %signal: !hal.fence
  ) {
    %a = hal.tensor.import wait(%wait) => %buf0 "input0"
        : !hal.buffer_view -> tensor<?x4xf32>{%dim0}
    %b = hal.tensor.import wait(%wait) => %buf1 "input1"
        : !hal.buffer_view -> tensor<?x4xf32>{%dim0}

    %result = func.call @compute(%a, %b)
        : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>

    %aliased = hal.tensor.alias wait(%wait) =>
        %result : tensor<?x4xf32>{%dim0} to %buf_out : !hal.buffer
    %annotated = hal.tensor.transients %aliased : tensor<?x4xf32>{%dim0}
        from %transients : !hal.buffer
    %ready = hal.tensor.barrier join(%annotated : tensor<?x4xf32>)
        => %signal : !hal.fence
    %out_bv = hal.tensor.export %ready "output0"
        : tensor<?x4xf32>{%dim0} -> !hal.buffer_view
    util.return
  }
}
"""
        ok, stderr = compile_mlir(mlir, "torch")
        self.assertTrue(ok, f"iree-compile failed:\n{stderr}")

    def test_v2_shared_offset_envelope(self):
        """V2: Shared parent buffer, dynamic offset, torch compute."""
        mlir = """\
module @pyre_kernel {
  func.func private @compute(
      %a_builtin: tensor<?xf32>, %b_builtin: tensor<?xf32>
  ) -> tensor<?xf32>
      attributes {inlining_policy = #util.inline.always} {
    %a = torch_c.from_builtin_tensor %a_builtin
        : tensor<?xf32> -> !torch.vtensor<[?], f32>
    %b = torch_c.from_builtin_tensor %b_builtin
        : tensor<?xf32> -> !torch.vtensor<[?], f32>
    %alpha = torch.constant.int 1
    %result = torch.aten.add.Tensor %a, %b, %alpha
        : !torch.vtensor<[?], f32>, !torch.vtensor<[?], f32>, !torch.int
        -> !torch.vtensor<[?], f32>
    %result_builtin = torch_c.to_builtin_tensor %result
        : !torch.vtensor<[?], f32> -> tensor<?xf32>
    return %result_builtin : tensor<?xf32>
  }

  util.func public @add_offset_envelope(
      %parent_buf: !hal.buffer_view,
      %off_b_raw: index,
      %count: index,
      %buf_out: !hal.buffer,
      %transients: !hal.buffer,
      %wait: !hal.fence,
      %signal: !hal.fence
  ) {
    %parent_elems = hal.buffer_view.dim<%parent_buf : !hal.buffer_view>[0] : index
    %parent = hal.tensor.import wait(%wait) => %parent_buf "parent"
        : !hal.buffer_view -> tensor<?xf32>{%parent_elems}

    %off_b = util.assume.int %off_b_raw<udiv = 4> : index
    %c0 = arith.constant 0 : index
    %a = tensor.extract_slice %parent[%c0] [%count] [1]
        : tensor<?xf32> to tensor<?xf32>
    %b = tensor.extract_slice %parent[%off_b] [%count] [1]
        : tensor<?xf32> to tensor<?xf32>

    %result = func.call @compute(%a, %b)
        : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>

    %aliased = hal.tensor.alias wait(%wait) =>
        %result : tensor<?xf32>{%count} to %buf_out : !hal.buffer
    %annotated = hal.tensor.transients %aliased : tensor<?xf32>{%count}
        from %transients : !hal.buffer
    %ready = hal.tensor.barrier join(%annotated : tensor<?xf32>)
        => %signal : !hal.fence
    %out_bv = hal.tensor.export %ready "output0"
        : tensor<?xf32>{%count} -> !hal.buffer_view
    util.return
  }
}
"""
        ok, stderr = compile_mlir(mlir, "torch")
        self.assertTrue(ok, f"iree-compile failed:\n{stderr}")

    def test_v3_inplace_envelope(self):
        """V3: In-place mutation, output aliases input buffer."""
        mlir = """\
module @pyre_kernel {
  func.func private @compute(
      %a_builtin: tensor<?x4xf32>, %b_builtin: tensor<?x4xf32>
  ) -> tensor<?x4xf32>
      attributes {inlining_policy = #util.inline.always} {
    %a = torch_c.from_builtin_tensor %a_builtin
        : tensor<?x4xf32> -> !torch.vtensor<[?,4], f32>
    %b = torch_c.from_builtin_tensor %b_builtin
        : tensor<?x4xf32> -> !torch.vtensor<[?,4], f32>
    %alpha = torch.constant.int 1
    %result = torch.aten.add.Tensor %a, %b, %alpha
        : !torch.vtensor<[?,4], f32>, !torch.vtensor<[?,4], f32>, !torch.int
        -> !torch.vtensor<[?,4], f32>
    %result_builtin = torch_c.to_builtin_tensor %result
        : !torch.vtensor<[?,4], f32> -> tensor<?x4xf32>
    return %result_builtin : tensor<?x4xf32>
  }

  util.func public @add_inplace_envelope(
      %buf_x: !hal.buffer_view,
      %buf_y: !hal.buffer_view,
      %buf_x_out: !hal.buffer,
      %dim0: index,
      %transients: !hal.buffer,
      %wait: !hal.fence,
      %signal: !hal.fence
  ) {
    %x = hal.tensor.import wait(%wait) => %buf_x "input_x"
        : !hal.buffer_view -> tensor<?x4xf32>{%dim0}
    %y = hal.tensor.import wait(%wait) => %buf_y "input_y"
        : !hal.buffer_view -> tensor<?x4xf32>{%dim0}

    %result = func.call @compute(%x, %y)
        : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>

    %aliased = hal.tensor.alias wait(%wait) =>
        %result : tensor<?x4xf32>{%dim0} to %buf_x_out : !hal.buffer
    %annotated = hal.tensor.transients %aliased : tensor<?x4xf32>{%dim0}
        from %transients : !hal.buffer
    %ready = hal.tensor.barrier join(%annotated : tensor<?x4xf32>)
        => %signal : !hal.fence
    %out_bv = hal.tensor.export %ready "output0"
        : tensor<?x4xf32>{%dim0} -> !hal.buffer_view
    util.return
  }
}
"""
        ok, stderr = compile_mlir(mlir, "torch")
        self.assertTrue(ok, f"iree-compile failed:\n{stderr}")

    def test_v5_mm_envelope(self):
        """V5: MM compute kernel with torch dialect."""
        mlir = """\
module @pyre_kernel {
  func.func private @compute(
      %mat1_builtin: tensor<?x?xf32>, %mat2_builtin: tensor<?x?xf32>
  ) -> tensor<?x?xf32>
      attributes {inlining_policy = #util.inline.always} {
    %mat1 = torch_c.from_builtin_tensor %mat1_builtin
        : tensor<?x?xf32> -> !torch.vtensor<[?,?], f32>
    %mat2 = torch_c.from_builtin_tensor %mat2_builtin
        : tensor<?x?xf32> -> !torch.vtensor<[?,?], f32>
    %result = torch.aten.mm %mat1, %mat2
        : !torch.vtensor<[?,?], f32>, !torch.vtensor<[?,?], f32>
        -> !torch.vtensor<[?,?], f32>
    %result_builtin = torch_c.to_builtin_tensor %result
        : !torch.vtensor<[?,?], f32> -> tensor<?x?xf32>
    return %result_builtin : tensor<?x?xf32>
  }

  util.func public @mm_envelope(
      %buf0: !hal.buffer_view,
      %buf1: !hal.buffer_view,
      %dim0_0: index,
      %dim0_1: index,
      %dim1_0: index,
      %dim1_1: index,
      %buf_out: !hal.buffer,
      %transients: !hal.buffer,
      %wait: !hal.fence,
      %signal: !hal.fence
  ) {
    %mat1 = hal.tensor.import wait(%wait) => %buf0 "input0"
        : !hal.buffer_view -> tensor<?x?xf32>{%dim0_0, %dim0_1}
    %mat2 = hal.tensor.import wait(%wait) => %buf1 "input1"
        : !hal.buffer_view -> tensor<?x?xf32>{%dim1_0, %dim1_1}

    %result = func.call @compute(%mat1, %mat2)
        : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %M = tensor.dim %result, %c0 : tensor<?x?xf32>
    %N = tensor.dim %result, %c1 : tensor<?x?xf32>

    %aliased = hal.tensor.alias wait(%wait) =>
        %result : tensor<?x?xf32>{%M, %N} to %buf_out : !hal.buffer
    %annotated = hal.tensor.transients %aliased : tensor<?x?xf32>{%M, %N}
        from %transients : !hal.buffer
    %ready = hal.tensor.barrier join(%annotated : tensor<?x?xf32>)
        => %signal : !hal.fence
    %out_bv = hal.tensor.export %ready "output0"
        : tensor<?x?xf32>{%M, %N} -> !hal.buffer_view
    util.return
  }
}
"""
        ok, stderr = compile_mlir(mlir, "torch")
        self.assertTrue(ok, f"iree-compile failed:\n{stderr}")

    def test_softmax_envelope(self):
        """Softmax compute kernel in envelope."""
        mlir = """\
module @pyre_kernel {
  func.func private @compute(
      %input_builtin: tensor<?x?xf32>
  ) -> tensor<?x?xf32>
      attributes {inlining_policy = #util.inline.always} {
    %input = torch_c.from_builtin_tensor %input_builtin
        : tensor<?x?xf32> -> !torch.vtensor<[?,?], f32>
    %dim = torch.constant.int 1
    %half_to_float = torch.constant.bool false
    %result = torch.aten._softmax %input, %dim, %half_to_float
        : !torch.vtensor<[?,?], f32>, !torch.int, !torch.bool
        -> !torch.vtensor<[?,?], f32>
    %result_builtin = torch_c.to_builtin_tensor %result
        : !torch.vtensor<[?,?], f32> -> tensor<?x?xf32>
    return %result_builtin : tensor<?x?xf32>
  }

  util.func public @softmax_envelope(
      %buf0: !hal.buffer_view,
      %dim0: index,
      %dim1: index,
      %buf_out: !hal.buffer,
      %transients: !hal.buffer,
      %wait: !hal.fence,
      %signal: !hal.fence
  ) {
    %input = hal.tensor.import wait(%wait) => %buf0 "input0"
        : !hal.buffer_view -> tensor<?x?xf32>{%dim0, %dim1}

    %result = func.call @compute(%input)
        : (tensor<?x?xf32>) -> tensor<?x?xf32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %result, %c0 : tensor<?x?xf32>
    %d1 = tensor.dim %result, %c1 : tensor<?x?xf32>

    %aliased = hal.tensor.alias wait(%wait) =>
        %result : tensor<?x?xf32>{%d0, %d1} to %buf_out : !hal.buffer
    %annotated = hal.tensor.transients %aliased : tensor<?x?xf32>{%d0, %d1}
        from %transients : !hal.buffer
    %ready = hal.tensor.barrier join(%annotated : tensor<?x?xf32>)
        => %signal : !hal.fence
    %out_bv = hal.tensor.export %ready "output0"
        : tensor<?x?xf32>{%d0, %d1} -> !hal.buffer_view
    util.return
  }
}
"""
        ok, stderr = compile_mlir(mlir, "torch")
        self.assertTrue(ok, f"iree-compile failed:\n{stderr}")


if __name__ == "__main__":
    run_tests()
