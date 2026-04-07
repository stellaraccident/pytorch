"""Tests for pyre compiler profile cache partitioning."""

import os
import pathlib
import re
import subprocess
import sys
import tempfile

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


_SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[2]
_ISA_PATTERN = re.compile("gfx" + r"\d{3,4}")


class TestCompilerCache(TestCase):
    def test_cache_namespaces_are_profile_partitioned(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            env = os.environ.copy()
            env["PYRE_CACHE_DIR"] = cache_dir
            if (
                "PYRE_IREE_COMPILER_CLI" not in env
                and "PYRE_IREE_COMPILE" in env
                and os.access(env["PYRE_IREE_COMPILE"], os.X_OK)
            ):
                env["PYRE_IREE_COMPILER_CLI"] = env.pop("PYRE_IREE_COMPILE")

            script = """
import torch
host = torch.empty(4, device='host:0')
host.fill_(1.0)
host64 = torch.empty(4, device='host:0', dtype=torch.float64)
host64.fill_(1.0)
try:
    gpu = torch.empty(4, device='hip:0')
    gpu.fill_(1.0)
    gpu64 = torch.empty(4, device='hip:0', dtype=torch.float64)
    gpu64.fill_(1.0)
    print("PYRE_HIP_FILL_OK")
except RuntimeError:
    pass
"""
            result = subprocess.run(
                [sys.executable, "-c", script],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            vmfbs = sorted(
                path.relative_to(pathlib.Path(cache_dir) / "kernels").as_posix()
                for path in (pathlib.Path(cache_dir) / "kernels").rglob("*.vmfb")
            )
            self.assertTrue(any(path.startswith("llvm-cpu-host/") for path in vmfbs))
            self.assertTrue(
                all("/" in path for path in vmfbs),
                msg=f"expected profile-prefixed cache paths, got {vmfbs}",
            )

            if "PYRE_HIP_FILL_OK" in result.stdout:
                self.assertTrue(
                    any(path.startswith("rocm-") for path in vmfbs),
                    msg=f"expected a rocm cache namespace, got {vmfbs}",
                )

    def test_no_hardcoded_gpu_isa_in_pyre_sources(self):
        roots = [
            _SOURCE_ROOT / "aten" / "src" / "ATen" / "pyre",
            _SOURCE_ROOT / "c10" / "pyre",
            _SOURCE_ROOT / "test" / "pyre",
        ]
        offenders = []
        for root in roots:
            for path in root.rglob("*"):
                if path == pathlib.Path(__file__).resolve() or not path.is_file():
                    continue
                if path.suffix not in {".cpp", ".h", ".py"}:
                    continue
                text = path.read_text()
                if _ISA_PATTERN.search(text):
                    offenders.append(str(path.relative_to(_SOURCE_ROOT)))

        self.assertEqual(
            offenders,
            [],
            msg=f"hardcoded GPU ISA literals found in {offenders}",
        )


if __name__ == "__main__":
    run_tests()
