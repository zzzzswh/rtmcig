"""The ``RTMCIG`` operator and the one-shot ``migrate`` function."""
from __future__ import annotations

import logging
import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from ._geometry import IGNORE_LOCATION, Survey
from ._imaging import (AngleBinner, OffsetImager, direction_angle, half_derivative, laplacian,
                       poynting_acoustic, poynting_scalar, reflection_angle, slant_stack)
from ._propagator import EQUATIONS, Propagator, check_deepwave_version, pad_model
from ._storage import SnapshotStore
from ._utils import (Timer, dtype_bytes, free_memory, human_bytes, is_numpy, like_input, logger,
                     resolve_device, resolve_dtype, to_tensor)
from ._wavelet import dominant_frequency, ricker

_METHODS = ("poynting", "offset")
_STORES = ("device", "cpu")
_INTERP = ("nearest", "hicks")


def _model_array(m, name, nx, nz, device, dtype):
    """``(nx, nz)`` array or scalar -> ``(nz, nx)`` tensor on the device."""
    if np.ndim(m) == 0:
        return torch.full((nz, nx), float(m), dtype=dtype, device=device)
    t = to_tensor(m, device=None, dtype=dtype)
    if t.ndim != 2:
        raise ValueError(f"{name} must be a scalar or an (nx, nz) array, got shape {tuple(t.shape)}")
    if t.shape != (nx, nz):
        raise ValueError(f"{name} must have shape (nx, nz) = ({nx}, {nz}), got {tuple(t.shape)}")
    return t.T.contiguous().to(device)


def _ensure_logging(verbose):
    if verbose and not logger.handlers and not logging.getLogger().handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[rtmcig] %(message)s"))
        logger.addHandler(h)
    if verbose:
        logger.setLevel(logging.INFO)


class RTMCIG:
    """Reverse-time migration of prestack data to angle-domain common-image
    gathers (ADCIGs), on the GPU through PyTorch and Deepwave.

    ``migrate`` maps data ``(ns, nr, nt)`` to gathers ``(nh, nx, nz)``: the
    reflection-angle axis first, then the image grid, the same layout as
    ``kirchcig``. ``.sum(0)`` of the gathers is the RTM image (exactly, for
    ``method="poynting"``), also available as ``op.image`` after a migration.

    Parameters
    ----------
    vel : (nx, nz) array or scalar, migration velocity [m/s] (a scalar needs
        ``nx`` and ``nz``)
    dx, dz : grid spacing [m]; the image grid is ``x = ox + ix*dx``,
        ``z = oz + iz*dz``
    srcs : (2, ns) source positions, rows (x, z) [m]
    recs : (2, nr) receiver positions shared by every shot, or (ns, 2, nr) a
        spread per shot. ``NaN`` marks a dead trace, which is ignored.
    nt, dt : trace length [samples] and sampling [s]
    wavelet : source wavelet, (nt,) or (ns, nt); or ``f0`` for a Ricker with
        that peak frequency and its peak at ``1/f0``
    f0 : peak frequency [Hz]; needed without ``wavelet``, otherwise estimated
        from it. Tunes the PML and the default Poynting smoothing.
    nh, hmax : number of angle bins and the largest reflection angle [deg].
        Bins are ``hmax/nh`` wide; ``angles`` gives their centres.
    signed : False (default): angles are ``|theta|`` in ``[0, hmax]``.
        True: ``[-hmax, hmax]``, positive when the source is on the -x side
        of the reflection point.
    method : "poynting" (default) or "offset"
        ``poynting``: Poynting-vector angle decomposition at every imaging
        time step (Yoon & Marfurt 2006). Cheap in memory and time; the
        gathers sum exactly to the image.
        ``offset``: subsurface-offset extended imaging condition followed by
        a slant stack to angle (Sava & Fomel 2003). Smoother gathers, better
        behaved where wavefields overlap, at the cost of ``2*offsets+1``
        multiply-adds per imaging step and an ``(nho, nx, nz)`` intermediate
        (``op.odcig``).
    equation : "scalar" (default) or "acoustic"
        ``scalar``: constant-density scalar wave equation (``deepwave.scalar``,
        regular grid, second order in time, ``accuracy``-order in space,
        C-PML). ``acoustic``: variable-density velocity-pressure equation on
        a staggered grid (``deepwave.acoustic``), needs ``rho``; the Poynting
        vector is then the physical energy flux ``p v``.
    rho : density [kg/m^3], scalar or (nx, nz), for ``equation="acoustic"``
    ox, oz : origin of the grid [m]
    image_dt : interval between imaging time steps [s], a multiple of ``dt``
        (default ``dt``). The cross-correlation only has to be sampled at
        the Nyquist rate of the data, so 2 to 4 ``dt`` is usually safe and
        divides the snapshot memory and imaging work by that factor.
    offsets, offset_step : ``method="offset"``: half-offsets from
        ``-offsets*offset_step*dx`` to ``+offsets*offset_step*dx`` in steps of
        ``offset_step`` cells (default 20 each side, every cell). The slant
        stack's angular resolution improves with the extent.
    offset_taper : fraction of the offset axis cosine-tapered at each end
        before the slant stack (default 0.2)
    poynting_smooth : half-width in cells of the moving average applied to
        the Poynting vectors before the angle is taken. Default: a quarter of
        the dominant wavelength (``min(vel) / (4 f0 dx)``). 0 disables.
    aperture : lateral migration aperture [m]: each shot batch is propagated
        only over the columns between the leftmost and rightmost of its
        sources and live receivers, extended by ``aperture`` on each side
        (None: the whole model). Shots propagated together share a window,
        so sort the shots along the line when using this.
    cig_x : x positions [m] at which to output gathers (default: every
        column). The image is always computed everywhere.
    shot_batch : shots propagated together, or "auto" (fit the snapshot
        storage into ``memory_fraction`` of the free device memory; on the
        CPU one shot per thread)
    store : where the source-wavefield snapshots live between the forward and
        the backward pass: "device" (default) or "cpu" (pinned host memory,
        streamed over PCIe on a side stream)
    store_dtype : "float16" (default, scaled per snapshot), "bfloat16" or
        "float32"
    memory_fraction : share of the free device memory ``shot_batch="auto"``
        may plan for (default 0.7)
    illumination : divide the gathers by the source illumination
        ``sum_t u_s^2`` stabilised with ``eps * max``; ``True`` uses
        ``eps = 0.01``, a float sets it, ``False`` (default) disables
    laplacian : apply ``-laplacian`` to the gathers and the image, the usual
        remedy for the low-wavenumber RTM artefacts (default False)
    halfderiv : apply the adjoint half-order time derivative (the 2D "rho
        filter", ``conj(sqrt(1 - rho e^{-iw}))``) to the data before
        migration (default False). In two dimensions a specular reflection
        is a single cylindrical wave, so its 45-degree phase and
        ``|w|^-1/2`` spectrum survive the cross-correlation and the plain 2D
        RTM wavelet is rotated by 45 degrees and tilted to low frequencies;
        the filter makes the imaged wavelet zero-phase and centred on the
        reflector. Same convention as ``kirchcig``'s ``halfderiv``.
    halfderiv_rho : leak of the half difference, default ``1 - 1/nt``
    pml_width : absorbing-layer width in cells (default 20)
    free_surface : rigid (pressure-free) top boundary instead of an absorbing
        one (default False). Use only when the data contain the surface
        multiples the propagation should reproduce.
    accuracy : spatial finite-difference order, 2, 4 (default), 6 or 8
    interpolation : "nearest" (default) rounds positions to the nearest cell;
        "hicks" injects/records at sub-cell positions with Kaiser-windowed
        sinc interpolation (Hicks 2002) of half-width ``hicks_halfwidth``
    device : "auto" (CUDA when available), "cuda", "cuda:1", "cpu", ...
    dtype : propagation precision, "float32" (default) or "float64"
    verbose : log progress per shot batch
    """

    def __init__(self, vel, dx, dz, srcs, recs, nt, dt, wavelet=None, f0=None,
                 nh=30, hmax=60.0, signed=False, method="poynting", equation="scalar", rho=None,
                 ox=0.0, oz=0.0, image_dt=None, offsets=20, offset_step=1, offset_taper=0.2,
                 poynting_smooth=None, aperture=None, cig_x=None, shot_batch="auto",
                 store="device", store_dtype="float16", memory_fraction=0.7,
                 illumination=False, laplacian=False, halfderiv=False, halfderiv_rho=None,
                 pml_width=20, free_surface=False, accuracy=4, interpolation="nearest",
                 hicks_halfwidth=4, device="auto", dtype="float32", verbose=False, nx=None, nz=None):
        check_deepwave_version()
        _ensure_logging(verbose)
        self.verbose = bool(verbose)
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype)
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be float32 or float64")

        # -- grid and models -----------------------------------------------------
        if np.ndim(vel) == 0:
            if nx is None or nz is None:
                raise ValueError("a scalar vel needs nx and nz")
            self.nx, self.nz = int(nx), int(nz)
        else:
            if np.ndim(vel) != 2:
                raise ValueError(f"vel must be (nx, nz), got shape {np.shape(vel)}")
            self.nx, self.nz = (int(s) for s in np.shape(vel))
            if nx is not None and nz is not None and (int(nx), int(nz)) != (self.nx, self.nz):
                raise ValueError("nx, nz disagree with vel.shape")
        self.dx, self.dz, self.ox, self.oz = float(dx), float(dz), float(ox), float(oz)
        if self.dx <= 0 or self.dz <= 0:
            raise ValueError("dx and dz must be positive")
        self._vel = _model_array(vel, "vel", self.nx, self.nz, self.device, self.dtype)
        if not torch.isfinite(self._vel).all() or float(self._vel.min()) <= 0:
            raise ValueError("vel must be finite and positive")
        if equation not in EQUATIONS:
            raise ValueError(f"equation must be one of {EQUATIONS}, got {equation!r}")
        self.equation = equation
        self._rho = None
        if equation == "acoustic":
            self._rho = _model_array(1000.0 if rho is None else rho, "rho", self.nx, self.nz,
                                     self.device, self.dtype)
        elif rho is not None:
            warnings.warn("rho is only used with equation='acoustic'", stacklevel=2)

        # -- survey ---------------------------------------------------------------
        self.survey = Survey(srcs, recs, self.nx, self.nz, self.dx, self.dz, self.ox, self.oz)
        self.ns, self.nr = self.survey.ns, self.survey.nr
        self.nt, self.dt = int(nt), float(dt)
        if self.nt < 2 or self.dt <= 0:
            raise ValueError("need nt >= 2 and dt > 0")

        # -- wavelet ----------------------------------------------------------------
        if wavelet is None:
            if f0 is None:
                raise ValueError("give either wavelet or f0 (peak frequency of a default Ricker)")
            w = ricker(float(f0), self.nt, self.dt)
        else:
            w = to_tensor(wavelet, device=None, dtype=torch.float32)
            if w.ndim not in (1, 2):
                raise ValueError("wavelet must be (nt,) or (ns, nt)")
            if w.ndim == 2 and w.shape[0] != self.ns:
                raise ValueError(f"a per-shot wavelet must be (ns, nt) with ns={self.ns}, got {tuple(w.shape)}")
            if w.shape[-1] > self.nt:
                raise ValueError(f"wavelet has {w.shape[-1]} samples, more than nt={self.nt}")
            if w.shape[-1] < self.nt:
                w = F.pad(w, (0, self.nt - w.shape[-1]))
        self._wavelet = w.contiguous()
        self.f0 = float(f0) if f0 is not None else dominant_frequency(w, self.dt)
        if self.f0 <= 0:
            raise ValueError("f0 must be positive")

        # -- angle axis ---------------------------------------------------------------
        self.nh, self.hmax, self.signed = int(nh), float(hmax), bool(signed)
        if self.nh < 1 or not (0 < self.hmax <= 90.0):
            raise ValueError("need nh >= 1 and 0 < hmax <= 90")
        self._binner = AngleBinner(self.nh, -self.hmax if self.signed else 0.0, self.hmax)
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
        self.method = method

        # -- offset axis (method="offset") ---------------------------------------------
        self.offsets, self.offset_step = int(offsets), int(offset_step)
        if self.offsets < 1 or self.offset_step < 1:
            raise ValueError("need offsets >= 1 and offset_step >= 1")
        self.offset_taper = float(offset_taper)
        self._h_cells = torch.arange(-self.offsets, self.offsets + 1, dtype=torch.int64) * self.offset_step
        self.nho = int(self._h_cells.numel())

        # -- imaging interval --------------------------------------------------------------
        if image_dt is None:
            self.image_interval = 1
        else:
            L = int(round(float(image_dt) / self.dt))
            if L < 1 or abs(L * self.dt - float(image_dt)) > 1e-6 * self.dt:
                raise ValueError("image_dt must be a positive integer multiple of dt")
            self.image_interval = L
        self.image_dt = self.image_interval * self.dt

        # -- Poynting smoothing, aperture, CIG columns -----------------------------------------
        if poynting_smooth is None:
            wavelength = float(self._vel.min()) / self.f0
            poynting_smooth = max(1, int(round(wavelength / (4.0 * self.dx))))
        self.poynting_smooth = int(poynting_smooth)
        if aperture is not None and float(aperture) <= 0:
            raise ValueError("aperture must be positive (metres) or None")
        self.aperture = None if aperture is None else float(aperture)
        self._aperture_cells = None if aperture is None else int(math.ceil(self.aperture / self.dx))
        if cig_x is None:
            self._cig_cols = None
            self.cig_x = self.x
        else:
            cols = np.rint((np.asarray(cig_x, dtype=np.float64).ravel() - self.ox) / self.dx).astype(np.int64)
            if cols.size == 0 or cols.min() < 0 or cols.max() >= self.nx:
                raise ValueError("cig_x positions must lie on the model")
            cols = np.unique(cols)
            self._cig_cols = torch.as_tensor(cols)
            self.cig_x = self.ox + cols * self.dx
        self.n_cig = int(len(self.cig_x))

        # -- batching, storage, filters -----------------------------------------------------------
        if shot_batch != "auto" and int(shot_batch) < 1:
            raise ValueError("shot_batch must be a positive int or 'auto'")
        self.shot_batch = shot_batch if shot_batch == "auto" else int(shot_batch)
        if store not in _STORES:
            raise ValueError(f"store must be one of {_STORES}")
        self.store = store
        self.store_dtype = resolve_dtype(store_dtype)
        if self.store_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("store_dtype must be float32, float16 or bfloat16")
        self.memory_fraction = float(memory_fraction)
        if illumination is True:
            illumination = 0.01
        self.illumination = None if illumination is False or illumination is None else float(illumination)
        self.laplacian = bool(laplacian)
        self.halfderiv = bool(halfderiv)
        self.halfderiv_rho = halfderiv_rho

        # -- propagation settings --------------------------------------------------------------------
        self.pml_width = int(pml_width)
        self.free_surface = bool(free_surface)
        if int(accuracy) not in (2, 4, 6, 8):
            raise ValueError("accuracy must be 2, 4, 6 or 8")
        self.accuracy = int(accuracy)
        if interpolation not in _INTERP:
            raise ValueError(f"interpolation must be one of {_INTERP}")
        self.interpolation = interpolation
        self.hicks_halfwidth = int(hicks_halfwidth)
        hw = self.hicks_halfwidth if interpolation == "hicks" else 0
        self._pad = (0 if self.free_surface else hw, hw, hw, hw)      # top, bottom, left, right
        self.max_vel = float(self._vel.max())

        # results of the last migration
        self.image = None
        self.illumination_map = None
        self.odcig = None

    # ------------------------------------------------------------------ axes
    @property
    def x(self):
        return self.ox + np.arange(self.nx) * self.dx

    @property
    def z(self):
        return self.oz + np.arange(self.nz) * self.dz

    @property
    def angles(self):
        """Bin centres of the angle axis [deg]."""
        return self._binner.centres().numpy()

    @property
    def h_axis(self):
        """Subsurface half-offsets of ``op.odcig`` [m] (``method="offset"``)."""
        return self._h_cells.numpy() * self.dx

    @property
    def shape_model(self):
        return (self.nh, self.n_cig, self.nz)

    @property
    def shape_data(self):
        return (self.ns, self.nr, self.nt)

    @property
    def wavelet(self):
        return self._wavelet.numpy()

    @property
    def vel(self):
        return self._vel.T.cpu().numpy()

    # -------------------------------------------------------------- planning
    def _time_plan(self):
        """``(K, nt_prop)``: the reversed-time receiver run is the exact adjoint
        of the forward run when reversed step ``m`` is paired with physical
        step ``nt_prop - 1 - m``, so the traces are zero-padded to
        ``nt_prop = K*L + 1`` and snapshots are taken at steps ``L, ..., (K-1)L``."""
        L = self.image_interval
        K = max(-(-(self.nt - 1) // L), 2)
        return K, K * L + 1

    @property
    def n_snapshots(self) -> int:
        return self._time_plan()[0] - 1

    def snapshot_bytes(self, shots: int = 1, ncols: int | None = None) -> int:
        """Storage of the source-wavefield snapshots for ``shots`` shots."""
        ncols = self.nx if ncols is None else ncols
        per_cell = SnapshotStore.bytes_per_cell(self.store_dtype, self.method == "poynting")
        return self.n_snapshots * shots * self.nz * ncols * per_cell

    def _work_bytes(self, shots: int, ncols: int) -> int:
        """Rough working set of one shot batch beyond the snapshots."""
        fd = self.accuracy // 2
        nzp = self.nz + sum(self._pad[:2]) + 2 * (self.pml_width + fd)
        nxp = ncols + sum(self._pad[2:]) + 2 * (self.pml_width + fd)
        fields = 8 if self.equation == "scalar" else 12
        b = dtype_bytes(self.dtype)
        work = shots * (fields * nzp * nxp * b + 16 * self.nz * ncols * 4)
        work += shots * max(self.survey.nu_max, 1) * (self.nt + self.image_interval) * b * 2
        return int(work)

    def _plan_batch(self, ncols: int) -> tuple[int, str]:
        store = self.store
        if self.shot_batch != "auto":
            return min(int(self.shot_batch), self.ns), store
        if self.device.type != "cuda":
            return max(1, min(self.ns, torch.get_num_threads(), 8)), "device"
        free = free_memory(self.device)
        budget = self.memory_fraction * free
        per_shot = self._work_bytes(1, ncols) + (self.snapshot_bytes(1, ncols) if store == "device" else 0)
        if per_shot > budget and store == "device":
            logger.info("snapshots of one shot (%s) exceed the memory budget (%s): storing them on the host",
                        human_bytes(self.snapshot_bytes(1, ncols)), human_bytes(budget))
            store = "cpu"
            per_shot = self._work_bytes(1, ncols)
        return int(max(1, min(self.ns, budget // max(per_shot, 1)))), store

    # ----------------------------------------------------------- propagators
    def _propagator(self, ix0: int, ix1: int) -> Propagator:
        vel = pad_model(self._vel[:, ix0:ix1], self._pad)
        rho = None if self._rho is None else pad_model(self._rho[:, ix0:ix1], self._pad)
        pml = [0 if self.free_surface else self.pml_width, self.pml_width, self.pml_width, self.pml_width]
        return Propagator(self.equation, vel, rho, self.dz, self.dx, self.dt, pml, self.f0,
                          self.accuracy, self.max_vel, crop=self._pad)

    def _hicks(self, frac: torch.Tensor, shape):
        """``deepwave`` Hicks interpolator for fractional cells ``frac (B, n, 2)``."""
        from deepwave.location_interpolation import Hicks
        return Hicks(frac.to(torch.float32), halfwidth=self.hicks_halfwidth,
                     free_surfaces=[self.free_surface, False, False, False],
                     model_shape=[int(shape[0]), int(shape[1])], dtype=torch.float32)

    def _batch_sources(self, shots: slice, ix0: int, nt_prop: int, shape):
        """Source amplitudes ``(B, n, nt_prop)`` and cells ``(B, n, 2)`` of a shot batch."""
        B = shots.stop - shots.start
        w = self._wavelet if self._wavelet.ndim == 2 else self._wavelet[None].expand(B, -1)
        if self._wavelet.ndim == 2:
            w = w[shots]
        w = F.pad(w, (0, nt_prop - self.nt))[:, None, :]                         # (B, 1, nt_prop)
        pad_t, _, pad_l, _ = self._pad
        if self.interpolation == "nearest":
            src, _, _ = self.survey.batch(shots, "cpu", ix0=ix0 - pad_l, iz0=-pad_t)
            return w, src
        frac = torch.as_tensor(self.survey.src_frac[shots]).clone()             # (B, 2)
        frac[:, 0] += pad_t
        frac[:, 1] += pad_l - ix0
        hicks = self._hicks(frac[:, None, :], shape)
        return hicks.source(w), hicks.get_locations()

    def _batch_receivers(self, shots: slice, ix0: int, data: torch.Tensor | None, shape):
        """Receiver cells ``(B, n, 2)`` and, when ``data (B, nr, nt)`` is given,
        the amplitudes to inject ``(B, n, nt)`` (dead traces removed, receivers
        sharing a cell summed)."""
        B = shots.stop - shots.start
        pad_t, _, pad_l, _ = self._pad
        live = torch.as_tensor(self.survey.live[shots])
        if self.interpolation == "nearest":
            _, rec, slot = self.survey.batch(shots, "cpu", ix0=ix0 - pad_l, iz0=-pad_t)
            if data is None:
                return rec, None, slot
            d = data * live.to(data.device, data.dtype)[..., None]
            merged = torch.zeros(B, rec.shape[1], d.shape[-1], dtype=d.dtype, device=d.device)
            merged.scatter_add_(1, slot.to(d.device)[..., None].expand(-1, -1, d.shape[-1]), d)
            return rec, merged, slot
        # Hicks: one interpolator per shot (its live receivers), padded to a common
        # size. Deepwave's Hicks class works on the CPU, so the expansion does too.
        data_cpu = None if data is None else data.detach().cpu()
        locs, amps, hicks_list = [], [], []
        for b in range(B):
            lv = live[b]
            frac = torch.as_tensor(self.survey.rec_frac[shots.start + b][lv.numpy()]).clone()
            frac[:, 0] += pad_t
            frac[:, 1] += pad_l - ix0
            h = self._hicks(frac[None], shape)
            hicks_list.append(h)
            locs.append(h.get_locations()[0])
            if data_cpu is not None:
                amps.append(h.source(data_cpu[b][lv][None])[0])
        n = max(l.shape[0] for l in locs)
        rec = torch.full((B, n, 2), IGNORE_LOCATION, dtype=torch.int64)
        for b, l in enumerate(locs):
            rec[b, : l.shape[0]] = l
        if data is None:
            return rec, None, hicks_list
        out = torch.zeros(B, n, data.shape[-1], dtype=data.dtype)
        for b, a in enumerate(amps):
            out[b, : a.shape[0]] = a
        return rec, out.to(data.device), hicks_list

    # ---------------------------------------------------------------- migrate
    def migrate(self, data):
        """Migration: data ``(ns, nr, nt)`` -> angle gathers ``(nh, n_cig, nz)``.

        NumPy in, NumPy out; tensor in, tensor out (on the compute device).
        Also sets ``image`` (nx, nz), ``illumination_map`` (nx, nz) and, for
        ``method="offset"``, ``odcig`` (nho, n_cig, nz)."""
        ref = data
        d_all = to_tensor(data, device=None, dtype=torch.float32)
        if d_all.shape != self.shape_data:
            raise ValueError(f"data must have shape {self.shape_data}, got {tuple(d_all.shape)}")
        dev = self.device
        L = self.image_interval
        K, nt_prop = self._time_plan()
        n_snap = K - 1
        if self.halfderiv:
            d_all = half_derivative(d_all, self.halfderiv_rho, adjoint=True)
        pad_t, pad_b, pad_l, pad_r = self._pad
        nz = self.nz

        image = torch.zeros(nz, self.nx, dtype=torch.float32, device=dev)
        illum = torch.zeros(nz, self.nx, dtype=torch.float32, device=dev)
        poynting = self.method == "poynting"
        if poynting:
            adcig_flat = torch.zeros(self.nh * nz * self.n_cig, dtype=torch.float32, device=dev)
            panel = nz * self.n_cig
            odcig = None
        else:
            odcig = torch.zeros(self.nho, nz, self.n_cig, dtype=torch.float32, device=dev)
            imager = OffsetImager(self._h_cells)
        iz_col = torch.arange(nz, device=dev)[:, None]

        ncols = max(np.ptp(self.survey.window(slice(s, s + 1), self._aperture_cells)) for s in range(self.ns))
        B, store_loc = self._plan_batch(int(ncols))
        logger.info("%s: %d shots in batches of %d, %d imaging steps, snapshots %s per shot on %s",
                    self.method, self.ns, B, n_snap, human_bytes(self.snapshot_bytes(1)), store_loc)

        for b0 in range(0, self.ns, B):
            b1 = min(b0 + B, self.ns)
            shots = slice(b0, b1)
            nb = b1 - b0
            ix0, ix1 = self.survey.window(shots, self._aperture_cells)
            nxs = ix1 - ix0
            prop = self._propagator(ix0, ix1)
            shape = (nz + pad_t + pad_b, nxs + pad_l + pad_r)
            with Timer(dev) as tm:
                # ---- source side: forward propagation, snapshots -------------------
                store = SnapshotStore(n_snap, nb, nz, nxs, dev, self.store_dtype, store_loc,
                                      direction=poynting)
                src_amp, src_loc = self._batch_sources(shots, ix0, nt_prop, shape)

                def fwd(step, f, store=store):
                    k = step // L - 1
                    if k < 0 or k >= n_snap:        # step 0 is empty; step K*L pairs with u_r(0) = 0
                        return
                    if poynting:
                        if self.equation == "scalar":
                            pz, px = poynting_scalar(f["u"], f["u_t"], self.dz, self.dx, self.poynting_smooth)
                            u = f["u"]
                        else:
                            pz, px = poynting_acoustic(f["p"], f["vz"], f["vx"], self.poynting_smooth)
                            u = f["p"]
                        store.put(k, u, direction_angle(pz, px))
                    else:
                        store.put(k, f["u"] if self.equation == "scalar" else f["p"])

                prop.run(src_amp, src_loc, L, fwd)
                store.finish_writes()

                # ---- receiver side: reversed data ---------------------------------------
                d = d_all[shots].to(dev)
                rec_loc, rec_amp, _ = self._batch_receivers(shots, ix0, d, shape)
                rec_amp = F.pad(rec_amp, (0, nt_prop - self.nt)).flip(-1)

                # cells of this batch in the output panels
                if self._cig_cols is None:
                    cols_local = None
                    cell = (iz_col * self.n_cig + torch.arange(ix0, ix1, device=dev)[None, :]) if poynting else None
                    out_cols = slice(ix0, ix1)
                else:
                    sel = (self._cig_cols >= ix0) & (self._cig_cols < ix1)
                    cols_local = (self._cig_cols[sel] - ix0).to(dev)
                    out_idx = torch.nonzero(sel).ravel().to(dev)
                    cell = (iz_col * self.n_cig + out_idx[None, :]) if poynting else None
                    out_cols = out_idx
                cols_sel = None if self._cig_cols is None or cols_local.numel() == 0 else cols_local
                skip_gathers = self._cig_cols is not None and cols_local.numel() == 0

                def bwd(step, f, store=store, cell=cell):
                    k = K - step // L - 1
                    if k < 0 or k >= n_snap:
                        return
                    us, dir_s = store.get(k, prefetch=k - 1)
                    ur = f["u"] if self.equation == "scalar" else f["p"]
                    val = us * ur
                    val.neg_()          # polarity: a velocity increase images as a positive peak
                    image[:, ix0:ix1] += val.sum(0)
                    illum[:, ix0:ix1] += (us * us).sum(0)
                    if skip_gathers:
                        return
                    if poynting:
                        if self.equation == "scalar":
                            prz, prx = poynting_scalar(ur, f["u_t"], self.dz, self.dx, self.poynting_smooth)
                        else:
                            prz, prx = poynting_acoustic(ur, f["vz"], f["vx"], self.poynting_smooth)
                        theta = reflection_angle(dir_s, prz, prx)
                        if cols_sel is not None:
                            theta = theta.index_select(-1, cols_sel)
                            val = val.index_select(-1, cols_sel)
                        if not self.signed:
                            theta = theta.abs()
                        self._binner.add(adcig_flat, panel, cell, theta, val)
                    else:
                        imager.add(odcig, cols_sel, out_cols, us.neg(), ur)

                prop.run(rec_amp, rec_loc, L, bwd)
                del store
            logger.info("shots %d-%d: columns %d-%d, %.2fs", b0, b1 - 1, ix0, ix1 - 1, tm.elapsed)

        # ---- assemble -------------------------------------------------------------
        if poynting:
            adcig = adcig_flat.view(self.nh, nz, self.n_cig)
        else:
            adcig = slant_stack(odcig, self._h_cells.to(torch.float32) * self.dx, self.dz,
                                torch.as_tensor(self.angles, dtype=torch.float32), self.offset_taper)
            self.odcig = like_input(odcig.permute(0, 2, 1).contiguous(), ref)
        if self.illumination is not None:
            ill = illum if self._cig_cols is None else illum[:, self._cig_cols.to(dev)]
            adcig = adcig / (ill + self.illumination * illum.max())[None]
            image = image / (illum + self.illumination * illum.max())
        if self.laplacian:
            image = laplacian(image, self.dz, self.dx)
            if self._cig_cols is None:
                adcig = laplacian(adcig, self.dz, self.dx)
            else:   # columns are not neighbours: depth term only
                up = F.pad(adcig.unsqueeze(1), (0, 0, 1, 1), mode="replicate").squeeze(1)
                adcig = -(up[:, 2:, :] - 2 * adcig + up[:, :-2, :]) / self.dz ** 2
        self.image = like_input(image.T.contiguous(), ref)
        self.illumination_map = like_input(illum.T.contiguous(), ref)
        return like_input(adcig.permute(0, 2, 1).contiguous(), ref)

    __call__ = migrate

    # ------------------------------------------------------------------ model
    def model(self, vel=None, rho=None, wavelet=None, shot_batch=None, max_vel=None):
        """Forward modelling with this operator's geometry and wavelet (or
        ``wavelet``, (nt,) or (ns, nt)) in ``vel`` (nx, nz) (default: the
        migration velocity) -> data ``(ns, nr, nt)``. Dead traces are zero.
        NumPy out, unless ``vel`` is a tensor.

        ``max_vel`` sets the velocity Deepwave uses for its internal time
        step (default: the largest of the migration and modelling
        velocities). Data sets that will be subtracted from each other, for
        instance to remove the direct wave, should be modelled with the same
        ``max_vel`` so that they share the internal time step."""
        ref = vel if isinstance(vel, torch.Tensor) else np.zeros(0)
        v = self._vel if vel is None else _model_array(vel, "vel", self.nx, self.nz, self.device, self.dtype)
        r = self._rho
        if self.equation == "acoustic" and rho is not None:
            r = _model_array(rho, "rho", self.nx, self.nz, self.device, self.dtype)
        if wavelet is not None:
            saved = self._wavelet
            w = to_tensor(wavelet, device=None, dtype=torch.float32)
            if w.shape[-1] > self.nt:
                raise ValueError("wavelet longer than nt")
            self._wavelet = F.pad(w, (0, self.nt - w.shape[-1])).contiguous()
        try:
            pml = [0 if self.free_surface else self.pml_width] + [self.pml_width] * 3
            mv = max(self.max_vel, float(v.max())) if max_vel is None else float(max_vel)
            prop = Propagator(self.equation, pad_model(v, self._pad),
                              None if r is None else pad_model(r, self._pad),
                              self.dz, self.dx, self.dt, pml, self.f0, self.accuracy,
                              mv, crop=self._pad)
            shape = (self.nz + self._pad[0] + self._pad[1], self.nx + self._pad[2] + self._pad[3])
            B = shot_batch or (self.shot_batch if self.shot_batch != "auto" else
                               (4 if self.device.type == "cuda" else max(1, min(torch.get_num_threads(), 8))))
            out = torch.zeros(self.ns, self.nr, self.nt, dtype=torch.float32)
            live = torch.as_tensor(self.survey.live)
            for b0 in range(0, self.ns, B):
                b1 = min(b0 + B, self.ns)
                shots = slice(b0, b1)
                src_amp, src_loc = self._batch_sources(shots, 0, self.nt, shape)
                rec_loc, _, aux = self._batch_receivers(shots, 0, None, shape)
                rec = prop.record(src_amp, src_loc, rec_loc).to(torch.float32).cpu()       # (B, n, nt)
                if self.interpolation == "nearest":
                    slot = aux
                    d = torch.gather(rec, 1, slot[..., None].expand(-1, -1, self.nt))
                else:
                    d = torch.zeros(b1 - b0, self.nr, self.nt)
                    for b, h in enumerate(aux):
                        n = h.get_locations().shape[1]
                        lv = live[b0 + b]
                        d[b, lv] = h.receiver(rec[b: b + 1, :n])[0]
                d = d * live[shots, :, None].to(d.dtype)
                out[shots] = d
        finally:
            if wavelet is not None:
                self._wavelet = saved
        return like_input(out, ref) if isinstance(ref, torch.Tensor) else out.numpy()

    # ------------------------------------------------------------------- demo
    _DEMO = dict(nx=101, nz=61, dx=10.0, dz=10.0, ns=5, nr=51, nt=401, dt=0.002, f0=20.0,
                 v0=2000.0, v1=2500.0, z_reflector=300.0, nh=18, hmax=90.0)

    @classmethod
    def demo(cls, **overrides):
        """Small operator: a horizontal reflector at 300 m in a 1 km x 600 m
        constant-velocity model, 5 shots, 51 receivers. Keyword overrides for
        the demo geometry (``ns``, ``nr``, ``nt``, ``dt``, ``f0``, ...) and
        anything else is passed to the constructor."""
        p = dict(cls._DEMO)
        p.update({k: v for k, v in overrides.items() if k in p})
        extra = {k: v for k, v in overrides.items() if k not in p}
        vel = np.full((p["nx"], p["nz"]), p["v0"], dtype=np.float32)
        srcs = np.stack([np.linspace(0.2, 0.8, p["ns"]) * (p["nx"] - 1) * p["dx"], np.zeros(p["ns"])])
        recs = np.stack([np.linspace(0.0, 1.0, p["nr"]) * (p["nx"] - 1) * p["dx"], np.zeros(p["nr"])])
        op = cls(vel, p["dx"], p["dz"], srcs, recs, p["nt"], p["dt"], f0=p["f0"],
                 nh=p["nh"], hmax=p["hmax"], **extra)
        op._demo = dict(p)
        return op

    def demo_true_velocity(self):
        d = self._require_demo()
        v = np.full((self.nx, self.nz), d["v0"], dtype=np.float32)
        v[:, int(round(d["z_reflector"] / self.dz)):] = d["v1"]
        return v

    def demo_data(self):
        """Reflection data of the demo: modelled in the two-layer model, with the
        direct wave (modelled in the background) subtracted."""
        self._require_demo()
        v_true = self.demo_true_velocity()
        mv = float(max(v_true.max(), self.max_vel))
        return self.model(v_true, max_vel=mv) - self.model(max_vel=mv)

    def _require_demo(self):
        d = getattr(self, "_demo", None)
        if d is None:
            raise RuntimeError("demo_data() is only available on RTMCIG.demo()")
        return d

    def __repr__(self):
        return (f"RTMCIG(shape_model={self.shape_model}, shape_data={self.shape_data}, "
                f"method={self.method!r}, equation={self.equation!r}, nh={self.nh}, hmax={self.hmax:g}, "
                f"signed={self.signed}, image_dt={self.image_dt:g}, halfderiv={self.halfderiv}, "
                f"device={str(self.device)!r})")


# --------------------------------------------------------------------- migrate
def migrate(data, vel, srcs, recs, dt, dx, dz, wavelet=None, f0=None, nh=30, hmax=60.0,
            method="poynting", **kwargs):
    """One-shot RTM of prestack data to angle gathers.

    Parameters
    ----------
    data : (ns, nr, nt) prestack traces
    vel : (nx, nz) migration velocity [m/s] (or a scalar with ``nx``, ``nz``)
    srcs, recs : (2, ns), (2, nr) or (ns, 2, nr) positions, rows (x, z) [m]
    dt, dx, dz : sampling [s], [m], [m]
    wavelet, f0, nh, hmax, method, ... : see :class:`RTMCIG`

    Returns
    -------
    (nh, nx, nz) angle gathers; ``.sum(0)`` is the RTM image. NumPy in,
    NumPy out; tensor in, tensor out.
    """
    if np.ndim(data) != 3:
        raise ValueError("data must be (ns, nr, nt)")
    nt = np.shape(data)[-1]
    op = RTMCIG(vel, dx, dz, srcs, recs, nt, dt, wavelet=wavelet, f0=f0, nh=nh, hmax=hmax,
                method=method, **kwargs)
    return op.migrate(data)


def fold_angles(adcig, angles):
    """Fold a signed gather ``(nh, ...)`` with bin centres ``angles`` [deg]
    (symmetric about 0) onto ``|theta|``: returns ``(nh // 2, ...)`` and the
    positive angles."""
    angles = np.asarray(angles)
    n = len(angles) // 2
    if len(angles) % 2 or not np.allclose(angles[:n], -angles[::-1][:n]):
        raise ValueError("the angle axis must be symmetric about zero with an even number of bins")
    if isinstance(adcig, torch.Tensor):
        folded = adcig[n:] + adcig[:n].flip(0)
    else:
        folded = adcig[n:] + adcig[:n][::-1]
    return folded, angles[n:]
