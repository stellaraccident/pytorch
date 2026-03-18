"""Pyre backend — PrivateUse1 registered as 'host'.

Auto-registers when USE_PYRE was enabled at build time.
After import: torch.device("host:0") works, torch.host is the device module.
"""

import torch

from . import _device


def _is_compiled() -> bool:
    """Return true if compiled with USE_PYRE."""
    return hasattr(torch._C, "_has_pyre") and torch._C._has_pyre


if _is_compiled():
    torch.utils.rename_privateuse1_backend("host")
    torch._register_device_module("host", _device)
    torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)
