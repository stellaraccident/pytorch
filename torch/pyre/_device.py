"""Device module for the 'host' backend.

Registered as torch.host via torch._register_device_module.
Follows the same API surface as torch.cuda / torch.xpu.
Phase 0: stubs that report a single device.
"""

import torch


def is_available() -> bool:
    return True


def device_count() -> int:
    return 1


def current_device() -> int:
    return 0


def set_device(device) -> None:
    idx = torch.accelerator._get_device_index(device, optional=True)
    if idx != 0:
        raise ValueError(f"host device index must be 0, got {idx}")


class device:
    """Context-manager that selects a host device."""

    def __init__(self, device):
        self.idx = torch.accelerator._get_device_index(device, optional=True)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return False


class Stream:
    """Placeholder stream for the host backend."""

    def __init__(self, device=0, stream_id=0):
        self.device_index = device
        self.stream_id = stream_id


def synchronize(device=None) -> None:
    pass
