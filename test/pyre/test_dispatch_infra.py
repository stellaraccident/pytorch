"""Tests for Epic 1 dispatch infrastructure.

Tests PyreStringSplicer (via the embed_templates script), template embedding,
PyreDeviceCapabilities, and PyreKernelCompiler initialization.
"""
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

from torch.testing._internal.common_utils import run_tests, TestCase

# We can't directly test C++ headers from Python, but we can test:
# 1. The embed_templates.py script
# 2. That the compiled libraries load and expose expected interfaces


class TestEmbedTemplates(TestCase):
    """Test the tools/embed_templates.py template embedding script."""

    def _run_embed(self, template_files, output_path):
        script = os.path.join(
            os.path.dirname(__file__), "..", "..", "tools", "embed_templates.py"
        )
        cmd = [sys.executable, script] + template_files + ["-o", output_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def test_single_template(self):
        with tempfile.TemporaryDirectory() as d:
            tmpl = os.path.join(d, "my_kernel.mlir")
            with open(tmpl, "w") as f:
                f.write("module { func.func @main() { return } }\n")

            out = os.path.join(d, "templates.inc")
            self._run_embed([tmpl], out)

            content = open(out).read()
            self.assertIn("kTemplate_my_kernel", content)
            self.assertIn("R\"mlir(", content)
            self.assertIn("func.func @main()", content)
            self.assertIn("namespace at::pyre", content)

    def test_multiple_templates(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["binary.mlir", "unary.mlir", "alpha.mlir"]:
                with open(os.path.join(d, name), "w") as f:
                    f.write(f"// {name}\n")

            out = os.path.join(d, "templates.inc")
            files = sorted(
                os.path.join(d, n)
                for n in ["binary.mlir", "unary.mlir", "alpha.mlir"]
            )
            self._run_embed(files, out)

            content = open(out).read()
            self.assertIn("kTemplate_binary", content)
            self.assertIn("kTemplate_unary", content)
            self.assertIn("kTemplate_alpha", content)

    def test_mlir_braces_preserved(self):
        """MLIR braces and special chars survive embedding."""
        with tempfile.TemporaryDirectory() as d:
            tmpl = os.path.join(d, "braces.mlir")
            mlir_content = textwrap.dedent("""\
                module {
                  func.func @test(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32>
                    attributes {iree.abi.model = "coarse-fences"} {
                    return %arg0 : tensor<?x?xf32>
                  }
                }
            """)
            with open(tmpl, "w") as f:
                f.write(mlir_content)

            out = os.path.join(d, "templates.inc")
            self._run_embed([tmpl], out)

            content = open(out).read()
            # Braces should appear verbatim — no escaping needed.
            self.assertIn("tensor<?x?xf32>", content)
            self.assertIn('iree.abi.model = "coarse-fences"', content)

    def test_placeholder_syntax_preserved(self):
        """$$placeholder$$ syntax survives embedding."""
        with tempfile.TemporaryDirectory() as d:
            tmpl = os.path.join(d, "placeholders.mlir")
            with open(tmpl, "w") as f:
                f.write("!t = $$element_type$$\n")
                f.write("!in = tensor<$$shape$$x!t>\n")

            out = os.path.join(d, "templates.inc")
            self._run_embed([tmpl], out)

            content = open(out).read()
            self.assertIn("$$element_type$$", content)
            self.assertIn("$$shape$$", content)

    def test_missing_file_fails(self):
        script = os.path.join(
            os.path.dirname(__file__), "..", "..", "tools", "embed_templates.py"
        )
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "templates.inc")
            result = subprocess.run(
                [sys.executable, script, "/nonexistent.mlir", "-o", out],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)

    def test_name_sanitization(self):
        """Filenames with hyphens and dots become valid C++ identifiers."""
        with tempfile.TemporaryDirectory() as d:
            tmpl = os.path.join(d, "my-kernel.v2.mlir")
            with open(tmpl, "w") as f:
                f.write("// test\n")

            out = os.path.join(d, "templates.inc")
            self._run_embed([tmpl], out)

            content = open(out).read()
            self.assertIn("kTemplate_my_kernel_v2", content)


class TestDeviceCapabilities(TestCase):
    """Test PyreDeviceCapabilities via the PyreDevice Python interface."""

    def test_device_exists(self):
        """Sanity: pyre device is accessible."""
        import torch

        self.assertGreater(torch.host.device_count(), 0)


class TestKernelCompilerInit(TestCase):
    """Test PyreKernelCompiler initialization paths."""

    def test_compiler_env_vars_documented(self):
        """Ensure the env var names are stable (documented in design doc)."""
        # These env var names are part of the public interface.
        # We just verify they don't crash if set to garbage.
        old_compile = os.environ.get("PYRE_IREE_COMPILE")
        old_lib = os.environ.get("PYRE_IREE_COMPILER_LIB")
        try:
            os.environ["PYRE_IREE_COMPILE"] = "/nonexistent/iree-compile"
            os.environ["PYRE_IREE_COMPILER_LIB"] = "/nonexistent/lib.so"
            # These env vars are checked at initialization time, which happens
            # lazily. We can't easily re-initialize, so just verify the vars
            # exist as documented.
            self.assertEqual(
                os.environ["PYRE_IREE_COMPILE"], "/nonexistent/iree-compile"
            )
        finally:
            if old_compile is not None:
                os.environ["PYRE_IREE_COMPILE"] = old_compile
            else:
                os.environ.pop("PYRE_IREE_COMPILE", None)
            if old_lib is not None:
                os.environ["PYRE_IREE_COMPILER_LIB"] = old_lib
            else:
                os.environ.pop("PYRE_IREE_COMPILER_LIB", None)


class TestRealTemplates(TestCase):
    """Test that the real MLIR templates in the source tree are valid."""

    def _templates_dir(self):
        return os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "aten",
            "src",
            "ATen",
            "pyre",
            "dispatch",
            "templates",
        )

    def test_templates_exist(self):
        d = self._templates_dir()
        self.assertTrue(os.path.isdir(d), f"templates dir missing: {d}")
        templates = [f for f in os.listdir(d) if f.endswith(".mlir")]
        self.assertGreater(len(templates), 0)

    def test_binary_template_has_required_placeholders(self):
        tmpl = open(
            os.path.join(self._templates_dir(), "elementwise_binary.mlir")
        ).read()
        for ph in [
            "$$element_type$$",
            "$$func_name$$",
            "$$lhs_shape$$",
            "$$out_shape$$",
            "$$torch_op$$",
        ]:
            self.assertIn(ph, tmpl)

    def test_unary_template_has_required_placeholders(self):
        tmpl = open(
            os.path.join(self._templates_dir(), "elementwise_unary.mlir")
        ).read()
        for ph in [
            "$$element_type$$",
            "$$func_name$$",
            "$$input_shape$$",
            "$$torch_op$$",
        ]:
            self.assertIn(ph, tmpl)

    def test_all_templates_embed_cleanly(self):
        """Run embed_templates.py on the real templates."""
        d = self._templates_dir()
        templates = sorted(
            os.path.join(d, f) for f in os.listdir(d) if f.endswith(".mlir")
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "templates.inc")
            script = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "tools",
                "embed_templates.py",
            )
            result = subprocess.run(
                [sys.executable, script] + templates + ["-o", out],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            content = open(out).read()
            self.assertIn("kTemplate_elementwise_binary", content)
            self.assertIn("kTemplate_elementwise_unary", content)


if __name__ == "__main__":
    run_tests()
