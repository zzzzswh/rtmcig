"""Small helpers shared by the package: devices, NumPy/Torch conversion, logging."""
from __future__ import annotations

import logging
import math
import os
import time

import numpy as np
import torch

logger = logging.getLogger("rtmcig")

_TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
}


def resolve_device(device) -> torch.device:
    """``"auto"`` picks CUDA when it is available, otherwise the CPU."""
    if device is None or device == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(device)


def resolve_dtype(dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        try:
            return _TORCH_DTYPES[dtype]
        except KeyError as exc:
            raise ValueError(f"unknown dtype {dtype!r}; use one of {sorted(_TORCH_DTYPES)}") from exc
    if dtype in (np.float32, np.dtype("float32")):
        return torch.float32
    if dtype in (np.float64, np.dtype("float64")):
        return torch.float64
    raise ValueError(f"unsupported dtype {dtype!r}")


def is_numpy(x) -> bool:
    return isinstance(x, np.ndarray)


def to_tensor(x, device=None, dtype=torch.float32) -> torch.Tensor:
    """NumPy array, sequence or tensor -> tensor of ``dtype`` on ``device``.

    Tensors already matching are returned as they are (no copy)."""
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(np.asarray(x))
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    if device is not None and t.device != torch.device(device):
        t = t.to(device)
    return t


def like_input(out: torch.Tensor, ref):
    """Return ``out`` as NumPy when ``ref`` was NumPy, else as a tensor."""
    if is_numpy(ref) or ref is None:
        return out.detach().cpu().numpy()
    return out


def dtype_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def human_bytes(n: float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def free_memory(device: torch.device) -> int | None:
    """Free bytes on the device: CUDA memory for a CUDA device, available host
    memory for the CPU, or None when it cannot be determined."""
    if device.type == "cuda":
        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    if device.type == "cpu":
        return host_available_memory()
    return None


def host_available_memory() -> int | None:
    """Available host memory in bytes (Linux ``MemAvailable``, else the free
    physical pages where ``sysconf`` reports them), or None when unknown."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return None


class Timer:
    """``with Timer() as t: ...; t.elapsed``; synchronises CUDA when asked."""

    def __init__(self, device=None):
        self.device = device
        self.elapsed = 0.0

    def __enter__(self):
        if self.device is not None and torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.device is not None and torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)
        self.elapsed = time.perf_counter() - self._t0
        return False


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def deg2rad(x: float) -> float:
    return x * math.pi / 180.0
