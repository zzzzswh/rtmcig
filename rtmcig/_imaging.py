"""Imaging conditions and angle decomposition.

Everything here works on batches of wavefields ``(B, nz, nx)`` (B shots
propagated together) and accumulates into image arrays laid out ``(nz, nx)``
or ``(nh, nz, nx)``; the operator transposes to ``kirchcig``'s ``(nh, nx, nz)``
at the very end.

Poynting-vector angle gathers (Yoon & Marfurt 2006)
    The energy-flux direction of the scalar wavefield is
    ``P = -u_t grad(u)``; for the velocity-pressure (staggered grid) equation
    it is ``P = p v`` directly. With ``n_s`` the propagation direction of the
    source wavefield and ``n_r`` that of the back-propagated receiver
    wavefield (both pointing at the reflector), the opening angle ``2 theta``
    satisfies ``cos 2theta = n_s . n_r``. We use the signed form
    ``2 theta = atan2(n_s x n_r, n_s . n_r)`` which needs no normalisation and
    is positive when the source is on the left (smaller x) of the receiver.

Subsurface-offset gathers (Sava & Fomel 2003)
    ``I(x, z, h) = sum_t u_s(x - h, z, t) u_r(x + h, z, t)``; the angle gather
    is the slant stack ``A(x, z, theta) = sum_h I(x, z + h tan theta, h)`` with
    the sign chosen so that positive angles again mean "source on the left".
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# --------------------------------------------------------- half-derivative
def half_derivative(d: torch.Tensor, rho: float | None = None, adjoint: bool = True) -> torch.Tensor:
    """Half-order time derivative along the last axis (Madagascar's
    ``sf_halfint`` / kirchcig's ``halfderiv``): ``H(w) = sqrt(1 - rho e^{-iw})``,
    the square root of the backward difference, with the default leak
    ``rho = 1 - 1/nt``. ``adjoint=True`` applies ``conj(H)``, which is what the
    data need before a 2D migration: in two dimensions a specular reflection
    is one cylindrical Green's function whose ``-45`` degree phase and
    ``|w|^-1/2`` spectrum are not cancelled by the back-propagation, so the
    plain RTM wavelet is rotated by 45 degrees and tilted to low frequencies;
    ``conj(H)`` removes both for a single source-receiver pair. With a
    receiver spread the sum over the receivers (a stationary-phase integral
    along the line) adds roughly another 45 degrees, which a second
    application removes."""
    nt = d.shape[-1]
    n = 2 * nt
    rho = (1.0 - 1.0 / nt) if rho is None else float(rho)
    w = 2 * math.pi * torch.fft.rfftfreq(n, d=1.0, device=d.device, dtype=torch.float64)
    H = torch.sqrt(1.0 - rho * torch.exp(-1j * w))
    if adjoint:
        H = H.conj()
    D = torch.fft.rfft(d.to(torch.float64), n=n, dim=-1)
    out = torch.fft.irfft(D * H, n=n, dim=-1)[..., :nt]
    return out.to(d.dtype)


# ----------------------------------------------------------------- derivatives
def spatial_gradient(u: torch.Tensor, dz: float, dx: float):
    """Central-difference ``(du/dz, du/dx)`` of ``u (B, nz, nx)``, one-sided at
    the edges (replicate padding)."""
    up = F.pad(u.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
    gz = (up[:, 2:, 1:-1] - up[:, :-2, 1:-1]) * (0.5 / dz)
    gx = (up[:, 1:-1, 2:] - up[:, 1:-1, :-2]) * (0.5 / dx)
    return gz, gx


def box_smooth(p: torch.Tensor, radius: int) -> torch.Tensor:
    """``(2r+1)^2`` moving average of ``p (B, nz, nx)``; the identity for r = 0."""
    if radius <= 0:
        return p
    k = 2 * int(radius) + 1
    return F.avg_pool2d(p.unsqueeze(1), k, stride=1, padding=int(radius),
                        count_include_pad=False).squeeze(1)


def laplacian(img: torch.Tensor, dz: float, dx: float) -> torch.Tensor:
    """Five-point ``-laplacian`` of the last two axes ``(..., nz, nx)`` (the
    minus keeps the polarity of a reflector peak)."""
    shape = img.shape
    u = img.reshape(-1, 1, shape[-2], shape[-1])
    up = F.pad(u, (1, 1, 1, 1), mode="replicate")
    lap = ((up[:, :, 2:, 1:-1] - 2 * u + up[:, :, :-2, 1:-1]) / dz ** 2
           + (up[:, :, 1:-1, 2:] - 2 * u + up[:, :, 1:-1, :-2]) / dx ** 2)
    return (-lap).reshape(shape)


def staggered_to_nodes(vz: torch.Tensor, vx: torch.Tensor):
    """Particle velocities of Deepwave's staggered grid (``vz`` at ``z+1/2``,
    ``vx`` at ``x+1/2``) averaged onto the pressure nodes."""
    vzp = F.pad(vz.unsqueeze(1), (0, 0, 1, 0), mode="replicate").squeeze(1)
    vxp = F.pad(vx.unsqueeze(1), (1, 0, 0, 0), mode="replicate").squeeze(1)
    return 0.5 * (vzp[:, 1:, :] + vzp[:, :-1, :]), 0.5 * (vxp[:, :, 1:] + vxp[:, :, :-1])


# --------------------------------------------------------------------- poynting
def poynting_scalar(u: torch.Tensor, u_t: torch.Tensor, dz: float, dx: float,
                    smooth: int):
    """``P = -u_t grad(u)`` of a scalar wavefield, box-smoothed over
    ``smooth`` cells (energy weighted, since |P| ~ u_t^2)."""
    gz, gx = spatial_gradient(u, dz, dx)
    pz = box_smooth(-u_t * gz, smooth)
    px = box_smooth(-u_t * gx, smooth)
    return pz, px


def poynting_acoustic(p: torch.Tensor, vz: torch.Tensor, vx: torch.Tensor, smooth: int):
    """``P = p v`` of the velocity-pressure wavefield (velocities already on nodes)."""
    return box_smooth(p * vz, smooth), box_smooth(p * vx, smooth)


def direction_angle(pz: torch.Tensor, px: torch.Tensor) -> torch.Tensor:
    """Propagation direction as an angle from the +z axis, radians in (-pi, pi]."""
    return torch.atan2(px, pz)


def reflection_angle(dir_s: torch.Tensor, prz: torch.Tensor, prx: torch.Tensor) -> torch.Tensor:
    """Signed reflection angle [degrees] from the source direction angle and the
    receiver-side Poynting vector. Positive when the source is on the -x side."""
    psz = torch.cos(dir_s)
    psx = torch.sin(dir_s)
    cross = psx * prz - psz * prx
    dot = psz * prz + psx * prx
    return torch.atan2(cross, dot) * (0.5 * 180.0 / math.pi)


# ----------------------------------------------------------------- angle binning
class AngleBinner:
    """Accumulates image contributions into angle bins.

    The gather has ``nh`` bins between ``angle_min`` and ``angle_max`` degrees
    with centres ``angle_min + (i + 0.5) dh``. A contribution is split linearly
    between the two bins whose centres bracket its angle, so the gather sums
    exactly to the image; angles beyond the axis are dropped.

    The target is a flat float32 buffer of ``nh * nz * nx_out`` cells passed in
    by the caller; ``cell`` maps every cell of the current batch grid to its
    flat index in one angle panel (so aperture windows and CIG column
    selections are just different ``cell`` tables).
    """

    def __init__(self, nh: int, angle_min: float, angle_max: float):
        self.nh = int(nh)
        self.angle_min = float(angle_min)
        self.angle_max = float(angle_max)
        self.dh = (self.angle_max - self.angle_min) / self.nh

    def centres(self) -> torch.Tensor:
        return self.angle_min + (torch.arange(self.nh, dtype=torch.float64) + 0.5) * self.dh

    def add(self, target_flat: torch.Tensor, panel_size: int, cell: torch.Tensor,
            theta_deg: torch.Tensor, val: torch.Tensor) -> None:
        """``theta_deg`` and ``val`` are ``(B, *cell.shape)``."""
        eps = 1e-4 * self.dh
        inside = (theta_deg >= self.angle_min - eps) & (theta_deg <= self.angle_max + eps)
        pos = ((theta_deg - self.angle_min) / self.dh - 0.5).clamp_(0.0, float(self.nh - 1))
        i0 = torch.floor(pos)
        w1 = pos - i0
        i0 = i0.to(torch.int64)
        i1 = (i0 + 1).clamp_(max=self.nh - 1)
        v = torch.where(inside, val, torch.zeros((), dtype=val.dtype, device=val.device))
        idx0 = i0 * panel_size + cell
        idx1 = i1 * panel_size + cell
        target_flat.index_add_(0, idx0.reshape(-1), (v * (1.0 - w1)).reshape(-1))
        target_flat.index_add_(0, idx1.reshape(-1), (v * w1).reshape(-1))


# ------------------------------------------------------------ offset gathers
class OffsetImager:
    """Subsurface-offset extended imaging condition.

    ``h_cells`` are the horizontal half-offsets in cells (symmetric, e.g.
    ``-20, -19, ..., 20``); ``I[ih, :, c] += sum_b u_s[b, :, x_c - h] u_r[b, :, x_c + h]``
    for the output columns ``c`` (positions ``x_c`` in the batch grid).
    """

    def __init__(self, h_cells: torch.Tensor):
        self.h_cells = h_cells.to(torch.int64).cpu()
        self.hmax = int(self.h_cells.abs().max()) if self.h_cells.numel() else 0

    def add(self, target: torch.Tensor, cols: torch.Tensor | None, out_cols: torch.Tensor | slice,
            us: torch.Tensor, ur: torch.Tensor) -> None:
        """``target (nho, nz, nx_out)``; ``cols`` int64 batch-grid columns to
        image (None: all), ``out_cols`` where they go in ``target``."""
        hm = self.hmax
        nx = us.shape[-1]
        if hm > 0:
            us_p = F.pad(us, (hm, hm))
            ur_p = F.pad(ur, (hm, hm))
        else:
            us_p, ur_p = us, ur
        for ih, h in enumerate(self.h_cells.tolist()):
            if cols is None:
                prod = us_p[..., hm - h: hm - h + nx] * ur_p[..., hm + h: hm + h + nx]
            else:
                prod = us_p.index_select(-1, cols + (hm - h)) * ur_p.index_select(-1, cols + (hm + h))
            target[ih, :, out_cols] += prod.sum(0)


def offset_taper(nho: int, frac: float, device=None) -> torch.Tensor:
    """Cosine taper over the outer ``frac`` of the offset axis (both ends)."""
    if frac <= 0:
        return torch.ones(nho, device=device)
    n = int(round(frac * (nho // 2)))
    w = torch.ones(nho, device=device)
    if n > 0:
        ramp = 0.5 * (1 - torch.cos(torch.linspace(0, math.pi, n + 2, device=device)[1:-1]))
        w[:n] = ramp
        w[nho - n:] = ramp.flip(0)
    return w


def slant_stack(odcig: torch.Tensor, h: torch.Tensor, dz: float, angles_deg: torch.Tensor,
                taper_frac: float = 0.2, col_chunk: int = 256) -> torch.Tensor:
    """Offset gathers ``(nho, nz, ncol)`` -> angle gathers ``(nh, nz, ncol)``.

    ``A(z, theta) = sum_h w_h I(z + h tan theta, h) / sum_h w_h``, with
    ``w_h`` the cosine taper over the outer ``taper_frac`` of the offset
    axis, evaluated in the Fourier domain along z (exact interpolation), with
    the depth axis zero-padded so that the shifted traces do not wrap around.
    ``h`` are the offsets in metres, ``angles_deg`` the output angles
    (|theta| < 90). Normalising by the taper weight (not by the number of
    offsets) gives a perfectly focused event the amplitude of the image at
    its angle whatever the taper.
    """
    nho, nz, ncol = odcig.shape
    device = odcig.device
    h = h.to(device=device, dtype=torch.float32)
    tan = torch.tan(torch.deg2rad(angles_deg.to(device=device, dtype=torch.float32)))
    nh = tan.numel()
    hmax = float(h.abs().max()) if nho else 0.0
    tmax = float(tan.abs().max()) if nh else 0.0
    pad = int(math.ceil(hmax * tmax / dz)) + 1
    nzp = nz + 2 * pad
    w = offset_taper(nho, taper_frac, device=device)
    kz = 2 * math.pi * torch.fft.rfftfreq(nzp, d=dz).to(device=device, dtype=torch.float32)  # (nk,)
    # phase[k, t, ih] = exp(i kz[k] * h[ih] * tan[t])
    phase = torch.exp(1j * (kz[:, None, None] * tan[None, :, None] * h[None, None, :]))    # (nk, nh, nho) c64
    phase = phase * (w / w.sum().clamp_min(1e-12))[None, None, :]
    out = torch.empty(nh, nz, ncol, dtype=odcig.dtype, device=device)
    for c0 in range(0, ncol, col_chunk):
        c1 = min(c0 + col_chunk, ncol)
        blk = F.pad(odcig[:, :, c0:c1], (0, 0, pad, pad))                                 # (nho, nzp, nc)
        spec = torch.fft.rfft(blk, dim=1)                                                 # (nho, nk, nc)
        spec = spec.permute(1, 0, 2)                                                      # (nk, nho, nc)
        acc = torch.bmm(phase, spec)                                                      # (nk, nh, nc)
        acc = acc.permute(1, 0, 2)                                                        # (nh, nk, nc)
        full = torch.fft.irfft(acc, n=nzp, dim=1)                                         # (nh, nzp, nc)
        out[:, :, c0:c1] = full[:, pad: pad + nz, :]
    return out
