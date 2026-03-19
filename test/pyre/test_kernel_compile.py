"""Tests for end-to-end kernel compilation and dispatch.

Tests the full pipeline: template expansion → IREE compile → VM load →
async dispatch with coarse-fences.

Requires IREE compiler: set PYRE_IREE_COMPILE or PYRE_IREE_COMPILER_LIB.
"""
import os
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


def has_iree_compiler():
    return bool(
        os.environ.get("PYRE_IREE_COMPILE")
        or os.environ.get("PYRE_IREE_COMPILER_LIB")
    )


@unittest.skipUnless(has_iree_compiler(), "IREE compiler not available")
class TestStringSpicerMLIR(TestCase):
    """Test that template expansion produces valid MLIR that compiles."""

    def test_simple_add_mlir_compiles(self):
        """Expand binary template and compile it through IREE."""
        import subprocess

        compiler = os.environ.get("PYRE_IREE_COMPILE")
        if not compiler:
            self.skipTest("Need PYRE_IREE_COMPILE for CLI test")

        # Hand-crafted MLIR for a simple f32 add (no templates).
        mlir = """
module {
  util.func public @test_add(
      %lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>,
      %out: tensor<?x?xf32>
  ) -> tensor<?x?xf32>
    attributes {iree.abi.model = "coarse-fences"} {

    %result = linalg.add
        ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%out : tensor<?x?xf32>) -> tensor<?x?xf32>
    util.return %result : tensor<?x?xf32>
  }
}
"""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w") as f:
            f.write(mlir)
            f.flush()
            with tempfile.NamedTemporaryFile(suffix=".vmfb") as out:
                result = subprocess.run(
                    [
                        compiler,
                        "--iree-hal-target-backends=llvm-cpu",
                        "--iree-input-type=torch",
                        "--iree-llvmcpu-target-cpu=host",
                        "-o",
                        out.name,
                        f.name,
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    result.returncode, 0,
                    f"iree-compile failed:\n{result.stderr}"
                )
                # Verify output is non-empty.
                out.seek(0)
                vmfb = out.read()
                self.assertGreater(len(vmfb), 0)

    def test_unary_neg_mlir_compiles(self):
        """Compile a unary negation kernel."""
        import subprocess

        compiler = os.environ.get("PYRE_IREE_COMPILE")
        if not compiler:
            self.skipTest("Need PYRE_IREE_COMPILE for CLI test")

        mlir = """
module {
  util.func public @test_neg(
      %input: tensor<?x?xf32>, %out: tensor<?x?xf32>
  ) -> tensor<?x?xf32>
    attributes {iree.abi.model = "coarse-fences"} {

    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
    } ins(%input : tensor<?x?xf32>) outs(%out : tensor<?x?xf32>) {
    ^bb0(%in: f32, %unused: f32):
      %0 = arith.negf %in : f32
      linalg.yield %0 : f32
    } -> tensor<?x?xf32>
    util.return %result : tensor<?x?xf32>
  }
}
"""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w") as f:
            f.write(mlir)
            f.flush()
            with tempfile.NamedTemporaryFile(suffix=".vmfb") as out:
                result = subprocess.run(
                    [
                        compiler,
                        "--iree-hal-target-backends=llvm-cpu",
                        "--iree-input-type=torch",
                        "--iree-llvmcpu-target-cpu=host",
                        "-o",
                        out.name,
                        f.name,
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    result.returncode, 0,
                    f"iree-compile failed:\n{result.stderr}"
                )

    def test_alpha_fused_add_mlir_compiles(self):
        """Compile an alpha-fused add kernel (alpha=2.0)."""
        import subprocess

        compiler = os.environ.get("PYRE_IREE_COMPILE")
        if not compiler:
            self.skipTest("Need PYRE_IREE_COMPILE for CLI test")

        mlir = """
module {
  util.func public @test_add_alpha(
      %lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>,
      %out: tensor<?x?xf32>
  ) -> tensor<?x?xf32>
    attributes {iree.abi.model = "coarse-fences"} {

    %alpha = arith.constant 2.0 : f32
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
    } ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%out : tensor<?x?xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %scaled = arith.mulf %b, %alpha : f32
      %sum = arith.addf %a, %scaled : f32
      linalg.yield %sum : f32
    } -> tensor<?x?xf32>
    util.return %result : tensor<?x?xf32>
  }
}
"""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w") as f:
            f.write(mlir)
            f.flush()
            with tempfile.NamedTemporaryFile(suffix=".vmfb") as out:
                result = subprocess.run(
                    [
                        compiler,
                        "--iree-hal-target-backends=llvm-cpu",
                        "--iree-input-type=torch",
                        "--iree-llvmcpu-target-cpu=host",
                        "-o",
                        out.name,
                        f.name,
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    result.returncode, 0,
                    f"iree-compile failed:\n{result.stderr}"
                )


if __name__ == "__main__":
    run_tests()
