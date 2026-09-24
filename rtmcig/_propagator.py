"""Wave propagation through Deepwave, exposing the wavefield every few steps.

Deepwave's propagators accept a ``forward_callback`` that is invoked every
``callback_frequency`` (user) time steps with the current wavefields, without
leaving the compiled C/CUDA time loop between calls. That is all the access an
RTM needs, so this module is a thin adapter around two of them:

``equation="scalar"``
    ``deepwave.scalar``: constant-density scalar wave equation, second order
    in time, 2/4/6/8th order in space on a regular grid, C-PML absorbing
    boundaries (Pasalic & McGarry 2010). The callback receives ``u`` and its
    time derivative ``u_t`` (from the two time levels Deepwave keeps).
``equation="acoustic"``
    ``deepwave.acoustic``: variable-density velocity-pressure formulation on a
    staggered grid (pressure at the nodes, particle velocities at the half
    nodes), same orders and the same C-PML. The callback receives the pressure
    ``p`` and the particle velocities ``vz``, ``vx`` averaged onto the nodes.

Both produce the same result for a constant density; the scalar one is
cheaper (one field instead of three) and is the default.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

import deepwave

from ._imaging import staggered_to_nodes

EQUATIONS = ("scalar", "acoustic")


def check_deepwave_version(minimum: str = "0.0.27") -> None:
    ver = getattr(getattr(deepwave, "_version", None), "__version__", None)
    if ver is None:
        return
    def key(v):
        return tuple(int(p) for p in v.split(".") if p.isdigit())
    if key(ver) < key(minimum):
        raise ImportError(f"rtmcig needs deepwave >= {minimum} (forward_callback support), found {ver}")


class Propagator:
    """One shot batch propagated through Deepwave with wavefield callbacks.

    Parameters
    ----------
    equation : "scalar" or "acoustic"
    vel : ``(nz, nx)`` float tensor on the compute device (already padded if
        the caller pads for sub-cell interpolation)
    rho : ``(nz, nx)`` density for the acoustic equation (None: 1000 kg/m^3)
    dz, dx, dt : grid spacing [m] and the *data* time step [s]; Deepwave
        subdivides ``dt`` internally to satisfy the CFL condition and
        resamples the source amplitudes accordingly
    pml_width : four PML widths in cells (top, bottom, left, right)
    pml_freq : dominant frequency the PML is tuned for [Hz]
    accuracy : spatial finite-difference order, 2/4/6/8
    max_vel : velocity used for the CFL condition (keeps forward and backward
        runs on the same internal time step)
    crop : ``(top, bottom, left, right)`` cells removed from the wavefields
        handed to the callback (internal padding of the model)
    """

    def __init__(self, equation: str, vel: torch.Tensor, rho: torch.Tensor | None,
                 dz: float, dx: float, dt: float, pml_width, pml_freq: float,
                 accuracy: int = 4, max_vel: float | None = None,
                 crop=(0, 0, 0, 0)):
        if equation not in EQUATIONS:
            raise ValueError(f"equation must be one of {EQUATIONS}, got {equation!r}")
        self.equation = equation
        self.vel = vel.contiguous()
        self.device = vel.device
        self.dtype = vel.dtype
        self.nz, self.nx = vel.shape
        if equation == "acoustic":
            if rho is None:
                rho = torch.full_like(vel, 1000.0)
            if rho.shape != vel.shape:
                raise ValueError(f"rho must have the velocity's shape {tuple(vel.shape)}, got {tuple(rho.shape)}")
            self.rho = rho.to(vel.device, vel.dtype).contiguous()
        else:
            self.rho = None
        self.dz, self.dx, self.dt = float(dz), float(dx), float(dt)
        self.pml_width = [int(p) for p in pml_width]
        self.pml_freq = float(pml_freq)
        self.accuracy = int(accuracy)
        self.max_vel = float(vel.abs().max()) if max_vel is None else float(max_vel)
        self.crop = tuple(int(c) for c in crop)
        self.inner_dt = deepwave.common.cfl_condition_n([self.dz, self.dx], self.dt, self.max_vel)[0]
        self.step_ratio = int(round(self.dt / self.inner_dt))
        # Deepwave's velocity-pressure engine adds dt*K*s(t) to p(t+1) (K = rho v^2 at the
        # source cell), i.e. it solves p_tt - v^2 lap p = K ds/dt, whereas the scalar engine
        # adds -v^2 dt^2 f(t) to u(t+1) and solves u_tt - v^2 lap u = -v^2 f. Injecting
        # s = -(1/rho) int f dt in the acoustic engine therefore makes its pressure the
        # scalar wavefield of the same amplitudes (exactly for a constant density), so
        # both engines record the same data and give images of the same phase, polarity
        # and scale.

    # ------------------------------------------------------------- helpers
    def _view(self, w: torch.Tensor) -> torch.Tensor:
        t, b, l, r = self.crop
        if any(self.crop):
            return w[:, t: w.shape[1] - b, l: w.shape[2] - r]
        return w

    def _common(self):
        return dict(pml_width=self.pml_width, pml_freq=self.pml_freq, accuracy=self.accuracy,
                    max_vel=self.max_vel)

    def _acoustic_amplitudes(self, a: torch.Tensor, locations: torch.Tensor) -> torch.Tensor:
        """``-(1/rho(x)) int a dt`` (trapezoidal), see the constructor."""
        integ = torch.cumsum(a, dim=-1) - 0.5 * a - 0.5 * a[..., :1]
        valid = locations[..., 0] != deepwave.common.IGNORE_LOCATION
        iz = locations[..., 0].clamp(0, self.nz - 1)
        ix = locations[..., 1].clamp(0, self.nx - 1)
        rho = torch.where(valid, self.rho[iz, ix], torch.ones((), dtype=self.dtype, device=self.device))
        return integ * (-self.dt / rho)[..., None]

    # ----------------------------------------------------------------- run
    @torch.no_grad()
    def run(self, amplitudes: torch.Tensor, locations: torch.Tensor, interval: int,
            callback: Callable[[int, dict], None]) -> None:
        """Propagate ``amplitudes (B, n, nt)`` injected at ``locations (B, n, 2)``
        (int64 cells ``(iz, ix)``, ``deepwave.IGNORE_LOCATION`` to skip),
        calling ``callback(step, fields)`` at user steps 0, interval, 2*interval, ...
        with the wavefields at that step (cropped ``(B, nz, nx)`` tensors)."""
        amplitudes = amplitudes.to(self.device, self.dtype).contiguous()
        locations = locations.to(self.device, torch.int64).contiguous()
        f32 = self.dtype != torch.float32     # imaging is done in float32
        if self.equation == "scalar":
            def cb(state):
                u = self._view(state.get_wavefield("wavefield_0"))
                um1 = self._view(state.get_wavefield("wavefield_m1"))
                u_t = (u - um1) * (1.0 / state.dt)
                if f32:
                    u, u_t = u.float(), u_t.float()
                callback(int(state.step), {"u": u, "u_t": u_t})
            deepwave.scalar(self.vel, (self.dz, self.dx), self.dt,
                            source_amplitudes=amplitudes, source_locations=locations,
                            forward_callback=cb, callback_frequency=int(interval), **self._common())
        else:
            def cb(state):
                p = self._view(state.get_wavefield("pressure_0"))
                vz, vx = staggered_to_nodes(self._view(state.get_wavefield("vy_0")),
                                            self._view(state.get_wavefield("vx_0")))
                if f32:
                    p, vz, vx = p.float(), vz.float(), vx.float()
                callback(int(state.step), {"p": p, "vz": vz, "vx": vx})
            deepwave.acoustic(self.vel, self.rho, (self.dz, self.dx), self.dt,
                              source_amplitudes_p=self._acoustic_amplitudes(amplitudes, locations),
                              source_locations_p=locations,
                              forward_callback=cb, callback_frequency=int(interval), **self._common())

    @torch.no_grad()
    def record(self, amplitudes: torch.Tensor, locations: torch.Tensor,
               receivers: torch.Tensor) -> torch.Tensor:
        """Forward modelling: data ``(B, nr, nt)`` recorded at ``receivers (B, nr, 2)``."""
        amplitudes = amplitudes.to(self.device, self.dtype).contiguous()
        locations = locations.to(self.device, torch.int64).contiguous()
        receivers = receivers.to(self.device, torch.int64).contiguous()
        if self.equation == "scalar":
            out = deepwave.scalar(self.vel, (self.dz, self.dx), self.dt,
                                  source_amplitudes=amplitudes, source_locations=locations,
                                  receiver_locations=receivers, **self._common())
            return out[-1]
        out = deepwave.acoustic(self.vel, self.rho, (self.dz, self.dx), self.dt,
                                source_amplitudes_p=self._acoustic_amplitudes(amplitudes, locations),
                                source_locations_p=locations,
                                receiver_locations_p=receivers, **self._common())
        return out[-3]          # receiver amplitudes are (p, vz, vx) in 2D


def pad_model(m: torch.Tensor, pad) -> torch.Tensor:
    """Replicate-pad ``(nz, nx)`` by ``(top, bottom, left, right)`` cells."""
    t, b, l, r = pad
    if not any(pad):
        return m
    return F.pad(m[None, None], (l, r, t, b), mode="replicate")[0, 0]
