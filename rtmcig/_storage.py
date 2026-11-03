"""Storage of the source wavefield snapshots between the forward and the
backward propagation.

The source wavefield is needed at every imaging time step while the receiver
wavefield is propagated backwards, so it is kept from the forward pass. Two
knobs control the footprint:

``dtype``
    ``float32`` (exact), ``float16`` (half the memory; every snapshot of every
    shot is scaled by its maximum before rounding, so the relative precision
    of the significant part of the wavefield is 1e-3) or ``bfloat16`` (half
    the memory, no scaling needed thanks to its exponent range, 3 significant
    digits).
``location``
    ``"device"`` keeps the snapshots next to the computation; ``"cpu"`` keeps
    them in pinned host memory and streams them over PCIe, overlapping the
    copies with the propagation on a side stream.

For the Poynting method the propagation direction of the source wavefield is
stored alongside as an int16 angle (2 bytes, 0.005 degree resolution), so a
Poynting snapshot in ``float16`` costs 4 bytes per cell, the same as the
plain ``float32`` wavefield of the offset method.
"""
from __future__ import annotations

import math

import torch

from ._utils import dtype_bytes

_Q = 32767.0 / math.pi   # radians -> int16


class SnapshotStore:
    def __init__(self, n_snap: int, batch: int, nz: int, nx: int, device,
                 dtype: torch.dtype = torch.float16, location: str = "device",
                 direction: bool = False):
        if location not in ("device", "cpu"):
            raise ValueError("location must be 'device' or 'cpu'")
        self.device = torch.device(device)
        if self.device.type == "cpu":
            location = "device"           # host storage *is* the device
        self.location = location
        self.dtype = dtype
        self.direction = direction
        self.n_snap, self.batch, self.nz, self.nx = int(n_snap), int(batch), int(nz), int(nx)
        self.scaled = dtype == torch.float16
        shape = (self.n_snap, self.batch, self.nz, self.nx)
        if location == "device":
            self._f = torch.empty(shape, dtype=dtype, device=self.device)
            self._d = torch.empty(shape, dtype=torch.int16, device=self.device) if direction else None
        else:
            self._f = torch.empty(shape, dtype=dtype, pin_memory=True)
            self._d = torch.empty(shape, dtype=torch.int16, pin_memory=True) if direction else None
            self._stream = torch.cuda.Stream(self.device)
            self._buf_f = [torch.empty(shape[1:], dtype=dtype, device=self.device) for _ in range(2)]
            self._buf_d = ([torch.empty(shape[1:], dtype=torch.int16, device=self.device) for _ in range(2)]
                           if direction else None)
            self._events = [None, None]
            self._buf_k = [-1, -1]
            self._put_events = []
        self._scale = (torch.ones((self.n_snap, self.batch), dtype=torch.float32, device=self.device)
                       if self.scaled else None)

    def free(self) -> None:
        """Release the snapshot memory now, even if something (a callback that
        Deepwave keeps alive in a reference cycle) still holds this object."""
        if self.location == "cpu" and hasattr(self, "_stream"):
            self._stream.synchronize()
        self._f = self._d = self._scale = None
        for name in ("_buf_f", "_buf_d", "_events", "_put_events"):
            if hasattr(self, name):
                setattr(self, name, None)

    # --------------------------------------------------------------- sizing
    @staticmethod
    def bytes_per_cell(dtype: torch.dtype, direction: bool) -> int:
        return dtype_bytes(dtype) + (2 if direction else 0)

    @property
    def nbytes(self) -> int:
        n = self._f.numel() * self._f.element_size()
        if self._d is not None:
            n += self._d.numel() * 2
        return n

    # ----------------------------------------------------------------- write
    def put(self, k: int, field: torch.Tensor, direction: torch.Tensor | None = None) -> None:
        """Store snapshot ``k``: ``field (B, nz, nx)`` float and, for the
        Poynting method, ``direction (B, nz, nx)`` in radians."""
        if self.scaled:
            amax = field.reshape(self.batch, -1).abs().amax(dim=1).clamp_min(1e-30)   # (B,)
            self._scale[k] = amax
            f = (field / amax[:, None, None]).to(self.dtype)
        else:
            f = field.to(self.dtype)
        d = None
        if self.direction:
            if direction is None:
                raise ValueError("this store expects a direction with every snapshot")
            d = (direction * _Q).round().to(torch.int16)
        if self.location == "device":
            self._f[k].copy_(f)
            if d is not None:
                self._d[k].copy_(d)
            return
        # host storage: copy on a side stream once the producing kernels are done.
        # The copy must read a private tensor: ``field`` may be a view of the
        # propagator's ping-pong buffer, which the next time steps overwrite.
        if f.data_ptr() == field.data_ptr():
            f = f.clone()
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._stream):
            self._stream.wait_event(ev)
            self._f[k].copy_(f, non_blocking=True)
            f.record_stream(self._stream)
            if d is not None:
                self._d[k].copy_(d, non_blocking=True)
                d.record_stream(self._stream)
            done = torch.cuda.Event()
            done.record(self._stream)
            self._put_events.append(done)

    def finish_writes(self) -> None:
        """Wait until every host copy has landed (call after the forward pass)."""
        if self.location == "cpu":
            self._stream.synchronize()
            self._put_events.clear()

    # ------------------------------------------------------------------ read
    def _issue_fetch(self, k: int, slot: int) -> None:
        with torch.cuda.stream(self._stream):
            self._buf_f[slot].copy_(self._f[k], non_blocking=True)
            if self._buf_d is not None:
                self._buf_d[slot].copy_(self._d[k], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self._stream)
        self._events[slot] = ev
        self._buf_k[slot] = k

    def get(self, k: int, prefetch: int | None = None):
        """Snapshot ``k`` as ``(field float32 (B, nz, nx), direction radians or None)``
        on the device. ``prefetch`` names the snapshot that will be asked for
        next, so a host store can start streaming it now."""
        if self.location == "device":
            f = self._f[k]
            d = self._d[k] if self._d is not None else None
        else:
            cur = torch.cuda.current_stream(self.device)
            slot = self._buf_k.index(k) if k in self._buf_k else None
            if slot is None:                       # not prefetched: fetch now
                slot = 0 if self._buf_k[1] == prefetch else 1
                self._stream.wait_stream(cur)      # kernels may still read this buffer
                self._issue_fetch(k, slot)
            cur.wait_event(self._events[slot])
            f = self._buf_f[slot]
            d = self._buf_d[slot] if self._buf_d is not None else None
            if prefetch is not None and 0 <= prefetch < self.n_snap and prefetch not in self._buf_k:
                other = 1 - slot
                if self._events[other] is not None:
                    # the other buffer may still be read by kernels on the compute stream
                    self._stream.wait_stream(cur)
                self._issue_fetch(prefetch, other)
        out = f.to(torch.float32)
        if self.scaled:
            out = out * self._scale[k][:, None, None]
        if d is not None:
            d = d.to(torch.float32) / _Q
        return out, d
