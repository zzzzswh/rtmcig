"""Acquisition geometry: positions in metres to grid cells.

Conventions follow ``kirchcig``: ``srcs`` is ``(2, ns)`` with rows ``(x, z)``,
``recs`` is ``(2, nr)`` (the same spread for every shot) or ``(ns, 2, nr)``
(a spread per shot). A receiver whose position is ``NaN`` is a dead trace:
it is ignored by the migration. The image grid is ``x = ox + ix*dx``,
``z = oz + iz*dz``; the propagation grid is the same grid stored as
``(nz, nx)`` because that is Deepwave's layout.
"""
from __future__ import annotations

import numpy as np
import torch

try:  # the constant lives in deepwave; keep a fallback so this module imports without it
    from deepwave.common import IGNORE_LOCATION
except Exception:  # pragma: no cover
    IGNORE_LOCATION = -1 << 31


def _positions(p, name: str) -> np.ndarray:
    """``(2, n)`` float64 array of ``(x, z)`` positions."""
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"{name} must be a (2, n) array of (x, z) positions, got shape {p.shape}")
    if p.shape[0] != 2:
        if p.shape[1] == 2:
            p = p.T
        else:
            raise ValueError(f"{name} must be a (2, n) array of (x, z) positions, got shape {p.shape}")
    return np.ascontiguousarray(p)


def _receivers(recs, ns: int) -> np.ndarray:
    """``(ns, 2, nr)`` float64 receiver positions; ``(2, nr)`` is broadcast to all shots."""
    r = np.asarray(recs, dtype=np.float64)
    if r.ndim == 2:
        r = _positions(r, "recs")
        return np.ascontiguousarray(np.broadcast_to(r[None], (ns, 2, r.shape[1])))
    if r.ndim == 3:
        if r.shape[0] != ns:
            raise ValueError(f"per-shot recs must have shape (ns, 2, nr) with ns={ns}, got {r.shape}")
        if r.shape[1] != 2:
            if r.shape[2] == 2:
                r = np.transpose(r, (0, 2, 1))
            else:
                raise ValueError(f"per-shot recs must have shape (ns, 2, nr), got {r.shape}")
        return np.ascontiguousarray(r)
    raise ValueError(f"recs must be (2, nr) or (ns, 2, nr), got shape {r.shape}")


class Survey:
    """Grid-cell bookkeeping for the sources and receivers of a survey.

    Parameters
    ----------
    srcs, recs : positions in metres, see the module docstring
    nx, nz, dx, dz, ox, oz : image / propagation grid

    Attributes
    ----------
    ns, nr : number of shots and receivers per shot (dead traces included)
    live : ``(ns, nr)`` bool, False for dead (NaN) traces
    src_cells : ``(ns, 2)`` int64 nearest cells ``(iz, ix)``
    rec_cells : ``(ns, nr, 2)`` int64 nearest cells, ``IGNORE_LOCATION`` for dead traces
    src_frac, rec_frac : the same positions as fractional cells (float64), for
        sub-cell (Hicks) interpolation
    rec_slot : ``(ns, nr)`` int64 index of the unique cell a receiver falls in
        (receivers sharing a cell are summed before injection; dead traces map
        to slot 0 with zero amplitude)
    rec_uniq : ``(ns, nu_max, 2)`` int64 unique receiver cells, ``IGNORE_LOCATION``-padded
    """

    def __init__(self, srcs, recs, nx: int, nz: int, dx: float, dz: float,
                 ox: float = 0.0, oz: float = 0.0):
        self.nx, self.nz = int(nx), int(nz)
        self.dx, self.dz, self.ox, self.oz = float(dx), float(dz), float(ox), float(oz)
        self.srcs = _positions(srcs, "srcs")
        self.ns = self.srcs.shape[1]
        self.recs = _receivers(recs, self.ns)
        self.nr = self.recs.shape[2]
        if not np.all(np.isfinite(self.srcs)):
            raise ValueError("source positions must be finite")

        self.live = np.all(np.isfinite(self.recs), axis=1)              # (ns, nr)
        if not self.live.any():
            raise ValueError("every receiver is dead (NaN position)")

        self.src_frac = np.stack([(self.srcs[1] - self.oz) / self.dz,
                                  (self.srcs[0] - self.ox) / self.dx], axis=-1)   # (ns, 2) (z, x)
        rec_fz = (self.recs[:, 1, :] - self.oz) / self.dz
        rec_fx = (self.recs[:, 0, :] - self.ox) / self.dx
        self.rec_frac = np.stack([rec_fz, rec_fx], axis=-1)                       # (ns, nr, 2)

        self._check_inside(self.src_frac.reshape(-1, 2), "srcs")
        self._check_inside(self.rec_frac[self.live], "recs")

        self.src_cells = np.rint(self.src_frac).astype(np.int64)
        rc = np.rint(np.nan_to_num(self.rec_frac, nan=0.0)).astype(np.int64)
        rc[~self.live] = IGNORE_LOCATION
        self.rec_cells = rc

        # -- receivers sharing a cell are merged: slot bookkeeping -------------
        slot = np.zeros((self.ns, self.nr), dtype=np.int64)
        n_uniq = np.zeros(self.ns, dtype=np.int64)
        uniq_list = []
        for s in range(self.ns):
            lv = self.live[s]
            flat = rc[s, lv, 0] * self.nx + rc[s, lv, 1]
            u, inv = np.unique(flat, return_inverse=True)
            slot[s, lv] = inv
            n_uniq[s] = u.size
            uniq_list.append(np.stack([u // self.nx, u % self.nx], axis=-1))
        self.n_uniq = n_uniq
        self.nu_max = int(n_uniq.max()) if self.ns else 0
        ru = np.full((self.ns, max(self.nu_max, 1), 2), IGNORE_LOCATION, dtype=np.int64)
        for s in range(self.ns):
            ru[s, : n_uniq[s]] = uniq_list[s]
        self.rec_uniq = ru
        self.rec_slot = slot
        self.n_merged = int(self.live.sum() - n_uniq.sum())

        # -- lateral extent of every shot (metres), for aperture windows --------
        xr = np.where(self.live, self.recs[:, 0, :], np.nan)
        self.x_min = np.fmin(self.srcs[0], np.nanmin(np.where(self.live, xr, np.inf), axis=1))
        self.x_max = np.fmax(self.srcs[0], np.nanmax(np.where(self.live, xr, -np.inf), axis=1))

    # ------------------------------------------------------------------ checks
    def _check_inside(self, frac: np.ndarray, name: str) -> None:
        if frac.size == 0:
            return
        bad = ((frac[:, 0] < -0.5) | (frac[:, 0] > self.nz - 0.5)
               | (frac[:, 1] < -0.5) | (frac[:, 1] > self.nx - 0.5))
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            x = frac[i, 1] * self.dx + self.ox
            z = frac[i, 0] * self.dz + self.oz
            raise ValueError(
                f"{int(bad.sum())} {name} position(s) fall outside the model, e.g. (x={x:g}, z={z:g}) m; "
                f"the grid covers x in [{self.ox:g}, {self.ox + (self.nx - 1) * self.dx:g}], "
                f"z in [{self.oz:g}, {self.oz + (self.nz - 1) * self.dz:g}] m")

    # ------------------------------------------------------------ batch access
    def window(self, shots: slice, aperture_cells: int | None) -> tuple[int, int]:
        """Column range ``[ix0, ix1)`` covering the sources and live receivers of
        ``shots`` plus ``aperture_cells`` on either side; the whole model when
        ``aperture_cells`` is None."""
        if aperture_cells is None:
            return 0, self.nx
        xmin = float(np.min(self.x_min[shots]))
        xmax = float(np.max(self.x_max[shots]))
        ix0 = int(np.floor((xmin - self.ox) / self.dx)) - int(aperture_cells)
        ix1 = int(np.ceil((xmax - self.ox) / self.dx)) + int(aperture_cells) + 1
        return max(ix0, 0), min(ix1, self.nx)

    def batch(self, shots: slice, device, ix0: int = 0, iz0: int = 0):
        """Nearest-cell tensors for a batch of shots, shifted to a sub-grid whose
        origin is column ``ix0`` (and row ``iz0``).

        Returns ``src (B, 1, 2)``, ``rec (B, nu_max, 2)`` (unique cells, padded
        with ``IGNORE_LOCATION``) and ``slot (B, nr)``, all int64 on ``device``.
        """
        src = torch.as_tensor(self.src_cells[shots]).clone()
        src[:, 0] -= iz0
        src[:, 1] -= ix0
        rec = torch.as_tensor(self.rec_uniq[shots]).clone()
        valid = rec[..., 0] != IGNORE_LOCATION
        rec[..., 0] = torch.where(valid, rec[..., 0] - iz0, rec[..., 0])
        rec[..., 1] = torch.where(valid, rec[..., 1] - ix0, rec[..., 1])
        slot = torch.as_tensor(self.rec_slot[shots])
        return src[:, None, :].to(device), rec.to(device), slot.to(device)

    def __repr__(self):
        return (f"Survey(ns={self.ns}, nr={self.nr}, dead={int((~self.live).sum())}, "
                f"merged={self.n_merged}, grid=({self.nx}x{self.nz}))")
