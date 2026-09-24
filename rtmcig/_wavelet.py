"""Source wavelets."""
from __future__ import annotations

import math

import torch


def ricker(f0: float, nt: int, dt: float, delay: float | None = None,
           dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Zero-phase Ricker wavelet with peak frequency ``f0`` [Hz], ``nt`` samples
    at ``dt`` [s], peak at ``delay`` [s] (default ``1/f0``, which makes it
    effectively causal). Returns a 1D tensor."""
    if f0 <= 0 or dt <= 0 or nt < 1:
        raise ValueError("need f0 > 0, dt > 0 and nt >= 1")
    if delay is None:
        delay = 1.0 / f0
    t = torch.arange(nt, dtype=torch.float64) * dt - delay
    a = (math.pi * f0 * t) ** 2
    return ((1.0 - 2.0 * a) * torch.exp(-a)).to(dtype)


def dominant_frequency(w: torch.Tensor, dt: float) -> float:
    """Frequency [Hz] of the amplitude-spectrum maximum of ``w`` (last axis).
    Used for the PML tuning frequency and the default Poynting smoothing."""
    w = torch.as_tensor(w, dtype=torch.float64).reshape(-1, w.shape[-1])
    n = max(w.shape[-1], 1024)
    spec = torch.fft.rfft(w, n=n).abs().sum(0)
    f = torch.fft.rfftfreq(n, dt)
    spec[0] = 0.0
    return float(f[int(spec.argmax())])
