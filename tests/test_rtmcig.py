"""Tests for rtmcig. Everything runs on the CPU in about a minute; the same
tests run on a GPU when one is available."""
import math
import warnings

import numpy as np
import pytest
import torch

import rtmcig
from rtmcig import RTMCIG, Survey, fold_angles, migrate, ricker

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------- fixtures
NX, NZ, DX = 81, 41, 10.0          # 800 m x 400 m
NT, DT, F0 = 301, 0.002, 20.0
Z0 = 195.0                          # interface between cells 19 and 20
V0, V1 = 2000.0, 2500.0


def geometry(ns=3, nr=41):
    vel = np.full((NX, NZ), V0, dtype=np.float32)
    srcs = np.stack([np.linspace(200.0, 600.0, ns), np.zeros(ns)])
    recs = np.stack([np.linspace(0.0, (NX - 1) * DX, nr), np.zeros(nr)])
    return vel, srcs, recs


def true_velocity():
    v = np.full((NX, NZ), V0, dtype=np.float32)
    v[:, 20:] = V1
    return v


def reflection_data(op):
    vt = true_velocity()
    return op.model(vt, max_vel=V1) - op.model(max_vel=V1)


def peak_depth(image, ix):
    col = image[ix]
    iz = int(np.abs(col[8:]).argmax()) + 8          # skip the near-surface artefacts
    return iz * DX


def event_angle(adcig, angles, ix, iz0=19, half=3):
    """Angle of the strongest event at the reflector: amplitude-weighted mean of
    the bins within 15 degrees of the peak bin."""
    g = np.abs(adcig[:, ix, iz0 - half: iz0 + half + 1]).sum(1)
    k = int(g.argmax())
    sel = np.abs(angles - angles[k]) <= 15.0
    return float((g[sel] * angles[sel]).sum() / g[sel].sum())


@pytest.fixture(scope="module")
def basic_op():
    vel, srcs, recs = geometry()
    return RTMCIG(vel, DX, DX, srcs, recs, NT, DT, f0=F0, nh=18, hmax=90.0, signed=True,
                  halfderiv=True, shot_batch=3)


@pytest.fixture(scope="module")
def basic_data(basic_op):
    return reflection_data(basic_op)


@pytest.fixture(scope="module")
def basic_result(basic_op, basic_data):
    adcig = basic_op.migrate(basic_data)
    return adcig, basic_op.image.copy(), basic_op.illumination_map.copy()


# ------------------------------------------------------------------ basics
def test_version_and_exports():
    assert isinstance(rtmcig.__version__, str)
    assert callable(migrate) and callable(rtmcig.cuda_available)


def test_shapes_axes_repr(basic_op, basic_data, basic_result):
    op = basic_op
    adcig, image, illum = basic_result
    assert basic_data.shape == op.shape_data == (3, 41, NT)
    assert adcig.shape == op.shape_model == (18, NX, NZ)
    assert image.shape == (NX, NZ) and illum.shape == (NX, NZ)
    assert adcig.dtype == np.float32
    assert np.allclose(op.angles, np.arange(-85.0, 90.0, 10.0))
    assert np.allclose(op.x, np.arange(NX) * DX) and np.allclose(op.z, np.arange(NZ) * DX)
    assert "RTMCIG" in repr(op) and "poynting" in repr(op)
    assert np.isfinite(adcig).all() and np.isfinite(image).all()
    assert (illum >= 0).all() and illum.max() > 0


def test_gathers_sum_to_image(basic_result):
    adcig, image, _ = basic_result
    assert np.abs(adcig.sum(0) - image).max() < 1e-5 * np.abs(image).max()


def test_reflector_depth_and_polarity(basic_op, basic_result):
    _, image, _ = basic_result
    for x in (300.0, 400.0, 500.0):
        ix = int(x / DX)
        assert abs(peak_depth(image, ix) - Z0) <= 1.5 * DX
        col = image[ix]
        assert col[8:][np.abs(col[8:]).argmax()] > 0    # velocity increase -> positive peak


def test_wavelet_is_zero_phase_at_zero_offset():
    """Single source-receiver pair at the same point, half-derivative on: the
    imaged wavelet is symmetric about the interface and positive."""
    vel = np.full((NX, NZ), V0, dtype=np.float32)
    srcs = np.array([[400.0], [0.0]])
    op = RTMCIG(vel, DX, DX, srcs, srcs, NT, DT, f0=F0, nh=9, hmax=90.0, halfderiv=True, shot_batch=1)
    op.migrate(reflection_data(op))
    col = op.image[40]
    k = 8 + int(np.abs(col[8:]).argmax())
    assert k in (19, 20)
    assert col[k] > 0
    # symmetry about z = 195 m: pairs (19, 20), (18, 21), (17, 22) close to each other
    for a, b in ((18, 21), (17, 22)):
        assert abs(col[a] - col[b]) < 0.3 * col[k]


def wavelet_phase(col, iz_skip=8, fine=16):
    """Phase [deg] of the analytic signal of an image column at its envelope
    peak: 0 for a zero-phase positive wavelet. The first ``iz_skip`` cells are
    zeroed to skip the near-surface artefacts."""
    c = np.array(col, dtype=np.float64)
    c[:iz_skip] = 0.0
    m = len(c)
    spec = np.fft.fft(c)
    a = np.zeros(m * fine, dtype=complex)          # analytic signal, upsampled
    a[0] = spec[0]
    a[1:(m + 1) // 2] = 2 * spec[1:(m + 1) // 2]
    if m % 2 == 0:
        a[m // 2] = spec[m // 2]
    z = np.fft.ifft(a) * fine
    return float(np.degrees(np.angle(z[int(np.abs(z).argmax())])))


def test_halfderiv_with_receiver_spread():
    """With a receiver line the sum over the receivers adds about 45 degrees
    that halfderiv does not remove; one more half-derivative of the data (a
    full derivative in total) does. See IMPLEMENTATION_NOTES section 3."""
    from rtmcig import half_derivative
    vel = np.full((NX, NZ), V0, dtype=np.float32)
    srcs = np.stack([np.linspace(200.0, 600.0, 17), np.zeros(17)])
    recs = np.stack([np.arange(NX) * DX, np.zeros(NX)])
    op = RTMCIG(vel, DX, DX, srcs, recs, NT, DT, f0=F0, nh=9, hmax=90.0, halfderiv=True,
                method="offset", offsets=1)
    data = reflection_data(op)
    op.migrate(data)
    half = wavelet_phase(op.image[38:43].mean(0))
    op.migrate(half_derivative(torch.as_tensor(data)).numpy())
    full = wavelet_phase(op.image[38:43].mean(0))
    assert 30.0 < half < 65.0           # measured +48
    assert abs(full) < 20.0             # measured +6
    assert abs(peak_depth(op.image, 40) - Z0) <= 1.5 * DX


# ----------------------------------------------------------------- angles
def angle_test_op(**kw):
    """One shot in the middle of a full-spread line."""
    vel = np.full((NX, NZ), V0, dtype=np.float32)
    srcs = np.array([[400.0], [0.0]])
    recs = np.stack([np.arange(NX) * DX, np.zeros(NX)])
    kw.setdefault("nh", 36)
    kw.setdefault("hmax", 90.0)
    kw.setdefault("signed", True)
    kw.setdefault("halfderiv", True)
    return RTMCIG(vel, DX, DX, srcs, recs, NT, DT, f0=F0, shot_batch=1, **kw)


@pytest.mark.parametrize("kw,tol", [
    (dict(method="poynting"), 8.0),
    (dict(method="offset", offsets=20), 3.0),
    (dict(method="poynting", equation="acoustic"), 8.0),
])
def test_angles_match_geometry(kw, tol):
    """The event angle follows the geometry. The offset method is accurate to
    a degree or two; the Poynting method is biased by several degrees for a
    reflector this shallow with a spread this long (crosstalk of the
    back-propagated receiver wavefield), which is its known limitation."""
    op = angle_test_op(**kw)
    adcig = op.migrate(reflection_data(op))
    for x in (250.0, 320.0, 400.0, 480.0, 550.0):
        ix = int(x / DX)
        expected = math.degrees(math.atan((x - 400.0) / Z0))
        got = event_angle(adcig, op.angles, ix)
        assert abs(got - expected) < tol, (x, expected, got)


def test_unsigned_is_fold_of_signed():
    op_s = angle_test_op(nh=36, signed=True)
    d = reflection_data(op_s)
    signed = op_s.migrate(d)
    op_u = angle_test_op(nh=18, signed=False)
    unsigned = op_u.migrate(d)
    folded, angles = fold_angles(signed, op_s.angles)
    assert np.allclose(angles, op_u.angles)
    assert np.abs(folded - unsigned).max() < 1e-4 * np.abs(unsigned).max()


def test_offset_method_outputs():
    op = angle_test_op(method="offset", offsets=10, offset_step=2)
    adcig = op.migrate(reflection_data(op))
    assert op.odcig.shape == (21, NX, NZ)
    assert np.allclose(op.h_axis, np.arange(-20, 21, 2) * DX)
    assert np.isfinite(adcig).all()
    # away from the shot the single-shot offset gather focuses at zero offset
    # (right below a shot it is a vertical strip over all offsets, as it should be)
    for ix in (28, 32, 48, 52):
        prof = np.abs(op.odcig[:, ix, 14:26]).sum(1)
        assert abs(int(prof.argmax()) - 10) <= 1


@pytest.mark.parametrize("taper", [0.0, 0.2, 0.5])
def test_slant_stack_keeps_focused_amplitude(taper):
    """A unit event along z = z0 + h tan(theta) stacks to amplitude 1 at theta,
    whatever the offset taper."""
    from rtmcig import slant_stack
    nz, dz = 200, 10.0
    h = torch.arange(-20, 21, dtype=torch.float32) * DX
    z = torch.arange(nz, dtype=torch.float32)[None, :] * dz
    for theta in (0.0, 20.0, 35.0):
        odcig = torch.exp(-0.5 * ((z - 1000.0 - h[:, None] * math.tan(math.radians(theta))) / 15.0) ** 2)
        a = slant_stack(odcig[:, :, None].contiguous(), h, dz, torch.tensor([theta]), taper)
        assert abs(float(a.max()) - 1.0) < 1e-3


# ----------------------------------------------------------- invariances
def test_shot_batch_invariance(basic_op, basic_data, basic_result):
    vel, srcs, recs = geometry()
    op1 = RTMCIG(vel, DX, DX, srcs, recs, NT, DT, f0=F0, nh=18, hmax=90.0, signed=True,
                 halfderiv=True, shot_batch=1)
    a1 = op1.migrate(basic_data)
    a3, _, _ = basic_result
    assert np.abs(a1 - a3).max() < 1e-4 * np.abs(a3).max()


def test_aperture(basic_data, basic_result):
    vel, srcs, recs = geometry()
    kw = dict(f0=F0, nh=18, hmax=90.0, signed=True, halfderiv=True, shot_batch=1)
    full = RTMCIG(vel, DX, DX, srcs, recs, NT, DT, aperture=5000.0, **kw).migrate(basic_data)
    ref, _, _ = basic_result
    assert np.abs(full - ref).max() < 1e-4 * np.abs(ref).max()
    op = RTMCIG(vel, DX, DX, srcs, recs, NT, DT, aperture=100.0, **kw)
    # receivers span the whole model, so the window is the full model anyway; use a
    # narrow spread instead to exercise the sub-model path
    recs_n = np.stack([np.linspace(250.0, 550.0, 31), np.zeros(31)])
    op = RTMCIG(vel, DX, DX, srcs, recs_n, NT, DT, aperture=100.0, **kw)
    d = reflection_data(op)
    a = op.migrate(d)
    assert a.shape == (18, NX, NZ)
    # shots at 200/400/600 m with receivers 250-550 m: windows [100, 650], [150, 650], [150, 700] m
    assert np.abs(a[:, :10]).max() == 0.0 and np.abs(a[:, 71:]).max() == 0.0   # outside every window
    assert np.abs(a[:, 12]).max() > 0
    assert abs(peak_depth(op.image, 40) - Z0) <= 1.5 * DX
    # the columns inside the windows agree with the full-model migration
    op_full = RTMCIG(vel, DX, DX, srcs, recs_n, NT, DT, **kw)
    a_full = op_full.migrate(d)
    inner = slice(30, 51)
    assert np.abs(a[:, inner] - a_full[:, inner]).max() < 0.05 * np.abs(a_full[:, inner]).max()


def test_cig_x_subset(basic_data, basic_result):
    vel, srcs, recs = geometry()
    ref, _, _ = basic_result
    xs = [200.0, 400.0, 605.0]                      # 605 rounds to column 60 (600 m)
    op = RTMCIG(vel, DX, DX, srcs, recs, NT, DT, f0=F0, nh=18, hmax=90.0, signed=True,
                halfderiv=True, shot_batch=3, cig_x=xs)
    a = op.migrate(basic_data)
    assert a.shape == (18, 3, NZ) and np.allclose(op.cig_x, [200.0, 400.0, 600.0])
    assert np.abs(a - ref[:, [20, 40, 60]]).max() < 1e-4 * np.abs(ref).max()
    assert op.image.shape == (NX, NZ)


@pytest.fixture(scope="module")
def result_fp32(basic_data):
    vel, srcs, recs = geometry()
    op = RTMCIG(vel, DX, DX, srcs, recs, NT, DT, f0=F0, nh=18, hmax=90.0, signed=True,
                halfderiv=True, shot_batch=3, store_dtype="float32")
    return op.migrate(basic_data)


@pytest.mark.parametrize("store_dtype,store,tol", [
    ("float32", "cpu", 1e-5), ("bfloat16", "device", 2e-2), ("float16", "cpu", 1e-3)])
def test_store_dtypes(basic_data, result_fp32, store_dtype, store, tol):
    """Compressed snapshot storage reproduces the float32 result (the default
    float16 store to better than 1e-3); the host store gives the same answer."""
    vel, srcs, recs = geometry()
    op = RTMCIG(vel, DX, DX, srcs, recs, NT, DT, f0=F0, nh=18, hmax=90.0, signed=True,
                halfderiv=True, shot_batch=3, store_dtype=store_dtype, store=store)
    a = op.migrate(basic_data)
    assert np.abs(a - result_fp32).max() < tol * np.abs(result_fp32).max()


def test_image_dt():
    vel, srcs, recs = geometry()
    kw = dict(f0=F0, nh=18, hmax=90.0, signed=True, halfderiv=True, shot_batch=3)
    op1 = RTMCIG(vel, DX, DX, srcs, recs, NT, DT, **kw)
    d = reflection_data(op1)
    a1 = op1.migrate(d)
    op2 = RTMCIG(vel, DX, DX, srcs, recs, NT, DT, image_dt=2 * DT, **kw)
    assert op2.image_interval == 2 and op2.n_snapshots < op1.n_snapshots
    a2 = op2.migrate(d)
    c = (a1 * a2).sum() / np.sqrt((a1 ** 2).sum() * (a2 ** 2).sum())
    assert c > 0.95
    assert abs(a2.sum(0)[40][8:].max() / a1.sum(0)[40][8:].max() - 0.5) < 0.1   # half the samples
    with pytest.raises(ValueError):
        RTMCIG(vel, DX, DX, srcs, recs, NT, DT, image_dt=1.5 * DT, **kw)


# ------------------------------------------------------------ geometry
def test_dead_traces_equal_zeroed_traces():
    vel, srcs, recs = geometry(ns=2, nr=21)
    kw = dict(f0=F0, nh=9, hmax=90.0, halfderiv=True, shot_batch=2)
    op = RTMCIG(vel, DX, DX, srcs, recs, NT, DT, **kw)
    d = reflection_data(op)
    recs_ps = np.broadcast_to(recs[None], (2, 2, 21)).copy()
    recs_ps[0, :, 3] = np.nan
    recs_ps[1, :, 10:15] = np.nan
    op_dead = RTMCIG(vel, DX, DX, srcs, recs_ps, NT, DT, **kw)
    assert op_dead.survey.live.sum() == 42 - 6
    d_zero = d.copy()
    d_zero[0, 3] = 0.0
    d_zero[1, 10:15] = 0.0
    a = op_dead.migrate(d)                 # amplitudes of dead traces are ignored
    b = op.migrate(d_zero)
    assert np.abs(a - b).max() < 1e-4 * np.abs(b).max()
    # modelling gives zeros on dead traces
    m = op_dead.model(true_velocity())
    assert np.abs(m[0, 3]).max() == 0.0 and np.abs(m[1, 10:15]).max() == 0.0 and np.abs(m[0, 4]).max() > 0


def test_duplicate_receivers_are_summed():
    vel, srcs, _ = geometry(ns=1)
    kw = dict(f0=F0, nh=9, hmax=90.0, halfderiv=True, shot_batch=1)
    # three receivers, two in the same cell (352 and 348 both round to 350 m)
    recs_dup = np.array([[100.0, 352.0, 348.0], [0.0, 0.0, 0.0]])
    recs_one = np.array([[100.0, 350.0, np.nan], [0.0, 0.0, np.nan]])
    op_dup = RTMCIG(vel, DX, DX, srcs, recs_dup, NT, DT, **kw)
    assert op_dup.survey.n_merged == 1 and op_dup.survey.nu_max == 2
    op_one = RTMCIG(vel, DX, DX, srcs, recs_one, NT, DT, **kw)
    rng = np.random.default_rng(0)
    d = rng.standard_normal((1, 3, NT)).astype(np.float32)
    d_sum = d.copy()
    d_sum[0, 1] = d[0, 1] + d[0, 2]
    d_sum[0, 2] = 0.0
    a = op_dup.migrate(d)
    b = op_one.migrate(d_sum)
    assert np.abs(a - b).max() < 1e-4 * np.abs(b).max()


def test_per_shot_receivers_and_survey_errors():
    vel, srcs, recs = geometry(ns=2, nr=11)
    recs_ps = np.stack([recs, recs + np.array([[5.0], [0.0]])])
    op = RTMCIG(vel, DX, DX, srcs, recs_ps, NT, DT, f0=F0)
    assert op.survey.recs.shape == (2, 2, 11)
    with pytest.raises(ValueError):
        Survey(np.array([[900.0], [0.0]]), recs, NX, NZ, DX, DX)      # outside the model
    with pytest.raises(ValueError):
        Survey(srcs, np.full((2, 3), np.nan), NX, NZ, DX, DX)          # everything dead
    with pytest.raises(ValueError):
        RTMCIG(vel, DX, DX, srcs, recs, NT, DT)                         # no wavelet, no f0
    with pytest.raises(ValueError):
        RTMCIG(vel, DX, DX, srcs, recs, NT, DT, f0=F0, method="nope")
    with pytest.raises(ValueError):
        op.migrate(np.zeros((2, 11, NT + 1), np.float32))


def test_hicks_interpolation():
    vel, srcs, _ = geometry(ns=2)
    recs = np.stack([np.linspace(3.0, 797.0, 41), np.full(41, 7.0)])   # off-grid positions
    kw = dict(f0=F0, nh=9, hmax=90.0, halfderiv=True, shot_batch=2)
    op_h = RTMCIG(vel, DX, DX, srcs, recs, NT, DT, interpolation="hicks", **kw)
    op_n = RTMCIG(vel, DX, DX, srcs, recs, NT, DT, **kw)
    vt = true_velocity()
    d_h = op_h.model(vt, max_vel=V1) - op_h.model(max_vel=V1)
    d_n = op_n.model(vt, max_vel=V1) - op_n.model(max_vel=V1)
    assert d_h.shape == d_n.shape and np.isfinite(d_h).all()
    c = (d_h * d_n).sum() / np.sqrt((d_h ** 2).sum() * (d_n ** 2).sum())
    assert c > 0.9
    a = op_h.migrate(d_h)
    assert a.shape == (9, NX, NZ) and np.isfinite(a).all()
    assert abs(peak_depth(op_h.image, 40) - Z0) <= 1.5 * DX


# -------------------------------------------------------------- options
def test_illumination_and_laplacian(basic_data):
    vel, srcs, recs = geometry()
    kw = dict(f0=F0, nh=18, hmax=90.0, signed=True, halfderiv=True, shot_batch=3)
    op = RTMCIG(vel, DX, DX, srcs, recs, NT, DT, illumination=True, laplacian=True, **kw)
    assert op.illumination == 0.01
    a = op.migrate(basic_data)
    assert np.isfinite(a).all()
    assert abs(peak_depth(op.image, 40) - Z0) <= 1.5 * DX
    op2 = RTMCIG(vel, DX, DX, srcs, recs, NT, DT, illumination=0.1, cig_x=[300.0, 400.0], laplacian=True, **kw)
    a2 = op2.migrate(basic_data)
    assert a2.shape == (18, 2, NZ) and np.isfinite(a2).all()


def test_acoustic_matches_scalar():
    vel, srcs, recs = geometry(ns=1)
    kw = dict(f0=F0, nh=18, hmax=90.0, signed=True, halfderiv=True, shot_batch=1)
    op_s = RTMCIG(vel, DX, DX, srcs, recs, NT, DT, **kw)
    op_a = RTMCIG(vel, DX, DX, srcs, recs, NT, DT, equation="acoustic", rho=1000.0, **kw)
    d_s = reflection_data(op_s)
    d_a = reflection_data(op_a)
    c = (d_s * d_a).sum() / np.sqrt((d_s ** 2).sum() * (d_a ** 2).sum())
    assert c > 0.99                                     # same data from both engines
    assert abs(np.abs(d_a).max() / np.abs(d_s).max() - 1) < 0.05
    a_s = op_s.migrate(d_s)
    i_s = op_s.image.copy()
    a_a = op_a.migrate(d_s)
    corr = lambda a, b: (a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum())
    assert corr(i_s, op_a.image) > 0.97                 # same image
    assert corr(a_s, a_a) > 0.9                         # same gathers, up to Poynting noise
    assert abs(peak_depth(op_a.image, 40) - Z0) <= 1.5 * DX


def test_wavelet_options():
    vel, srcs, recs = geometry(ns=2, nr=11)
    w = ricker(F0, 150, DT)
    op = RTMCIG(vel, DX, DX, srcs, recs, NT, DT, wavelet=w)
    assert op.wavelet.shape == (NT,) and abs(op.f0 - F0) < 3.0
    w2 = np.stack([ricker(F0, NT, DT).numpy(), ricker(25.0, NT, DT).numpy()])
    op2 = RTMCIG(vel, DX, DX, srcs, recs, NT, DT, wavelet=w2, f0=F0)
    d = op2.model(true_velocity())
    assert d.shape == (2, 11, NT) and np.isfinite(d).all()
    with pytest.raises(ValueError):
        RTMCIG(vel, DX, DX, srcs, recs, NT, DT, wavelet=np.zeros((3, NT)))


def test_torch_io_and_function(basic_op, basic_data, basic_result):
    ref, image, _ = basic_result
    d_t = torch.as_tensor(basic_data)
    a_t = basic_op.migrate(d_t)
    assert isinstance(a_t, torch.Tensor) and isinstance(basic_op.image, torch.Tensor)
    assert np.abs(a_t.cpu().numpy() - ref).max() < 1e-4 * np.abs(ref).max()
    vel, srcs, recs = geometry()
    a_f = migrate(basic_data, vel, srcs, recs, DT, DX, DX, f0=F0, nh=18, hmax=90.0, signed=True,
                  halfderiv=True, shot_batch=3)
    assert np.abs(a_f - ref).max() < 1e-4 * np.abs(ref).max()
    # scalar velocity with nx, nz
    a_c = migrate(basic_data, V0, srcs, recs, DT, DX, DX, f0=F0, nh=18, hmax=90.0, signed=True,
                  halfderiv=True, shot_batch=3, nx=NX, nz=NZ)
    assert np.abs(a_c - ref).max() < 1e-4 * np.abs(ref).max()


def test_demo():
    op = RTMCIG.demo(ns=2, nr=21, nt=201, shot_batch=2)
    d = op.demo_data()
    assert d.shape == (2, 21, 201)
    a = op.migrate(d)
    assert a.shape == (18, 101, 61) and np.isfinite(a).all()
    with pytest.raises(RuntimeError):
        RTMCIG(np.full((11, 11), 2000.0), 10.0, 10.0, [[50.0], [0.0]], [[50.0], [0.0]], 50, 0.002, f0=20.0).demo_data()


def test_planning_helpers(basic_op):
    op = basic_op
    assert op.n_snapshots == NT - 2
    assert op.snapshot_bytes(1) == op.n_snapshots * NZ * NX * 4       # float16 + int16 direction
    B, store = op._plan_batch(NX)
    assert 1 <= B <= 3 and store in ("device", "cpu")


def test_cpu_batch_respects_host_memory(monkeypatch):
    import rtmcig._operator as operator
    vel, srcs, recs = geometry(ns=6)
    op = RTMCIG(vel, DX, DX, srcs, recs, NT, DT, f0=F0, device="cpu", shot_batch="auto")
    per_shot = op._work_bytes(1, NX) + op.snapshot_bytes(1, NX)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
    monkeypatch.setattr(operator, "free_memory", lambda device: None)
    assert op._plan_batch(NX) == (6, "device")                  # unknown memory: one shot per thread
    monkeypatch.setattr(operator, "free_memory", lambda device: int(2.5 * per_shot / op.memory_fraction))
    assert op._plan_batch(NX) == (2, "device")
    monkeypatch.setattr(operator, "free_memory", lambda device: int(0.5 * per_shot / op.memory_fraction))
    with pytest.warns(UserWarning, match="host memory"):
        assert op._plan_batch(NX) == (1, "device")
