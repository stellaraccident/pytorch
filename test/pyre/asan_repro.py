"""Minimal repro for segfault in .to("host:0") — run under ASAN."""
import torch
print("Moving tensor to host:0...", flush=True)
x = torch.randn(256, 64).to("host:0")
print("OK", x.shape, flush=True)
