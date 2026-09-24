"""Layered model: model reflection data, migrate to angle gathers with both
methods and plot the image and a gather.

    python examples/layered_model.py            # ~6 min on one CPU core, seconds on a GPU
    python examples/layered_model.py --save docs/img/adcig.png
"""
import argparse
import time

import numpy as np

from rtmcig import RTMCIG

parser = argparse.ArgumentParser()
parser.add_argument("--save", default=None, help="write the figure to this file instead of showing it")
parser.add_argument("--ns", type=int, default=48, help="number of shots")
parser.add_argument("--device", default="auto")
args = parser.parse_args()

# ---- grid, model ----------------------------------------------------------
nx, nz, dx = 201, 101, 10.0                    # 2 km x 1 km
x, z = np.arange(nx) * dx, np.arange(nz) * dx
v0 = 2000.0
v_true = np.full((nx, nz), v0, np.float32)
v_true[:, z >= 400.0] = 2300.0                 # flat interface at 400 m
dip = 700.0 + 0.15 * x                         # dipping interface, 700 m at x=0 to 1000 m at x=2 km
v_true[z[None, :] >= dip[:, None]] = 2700.0
from scipy.ndimage import gaussian_filter
v_mig = gaussian_filter(v_true, sigma=8.0).astype(np.float32)   # smooth migration velocity

# ---- acquisition ------------------------------------------------------------
ns, nr = args.ns, 201
srcs = np.stack([np.linspace(300.0, 1700.0, ns), np.zeros(ns)])
recs = np.stack([x, np.zeros(nr)])
nt, dt, f0 = 1001, 0.002, 20.0

common = dict(nh=36, hmax=90.0, signed=True, halfderiv=True, laplacian=True,
              device=args.device, verbose=True)
op = RTMCIG(v_mig, dx, dx, srcs, recs, nt, dt, f0=f0, **common)
print(op)

# ---- data: reflections only (the direct wave, which travels in the top layer, is
#      modelled in the constant top-layer velocity and removed) --------------------
t0 = time.time()
mv = float(v_true.max())
data = op.model(v_true, max_vel=mv) - op.model(np.full_like(v_true, v0), max_vel=mv)
print(f"modelling {data.shape}: {time.time() - t0:.1f}s")

# ---- migration ---------------------------------------------------------------
t0 = time.time()
adcig_p = op.migrate(data)
image = op.image
print(f"poynting: {time.time() - t0:.1f}s")

op_o = RTMCIG(v_mig, dx, dx, srcs, recs, nt, dt, f0=f0, method="offset", offsets=30, **common)
t0 = time.time()
adcig_o = op_o.migrate(data)
print(f"offset: {time.time() - t0:.1f}s")

# ---- plot ------------------------------------------------------------------------
import matplotlib
if args.save:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

ix = 100                                       # gather at x = 1000 m
clip = 0.3 * np.abs(image[:, 20:]).max()
fig, ax = plt.subplots(1, 4, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[1.3, 1.3, 1, 1]))
ax[0].imshow(v_true.T, extent=[0, x[-1], z[-1], 0], cmap="viridis", aspect="auto")
ax[0].plot(srcs[0], srcs[1], "r*", ms=6)
ax[0].set(title="true velocity, shots", xlabel="x [m]", ylabel="z [m]")
ax[1].imshow(image.T, extent=[0, x[-1], z[-1], 0], cmap="gray", vmin=-clip, vmax=clip, aspect="auto")
ax[1].axvline(x[ix], color="c", lw=0.8)
ax[1].set(title="RTM image (stack of the gathers)", xlabel="x [m]")
for a, g, name in ((ax[2], adcig_p, "Poynting"), (ax[3], adcig_o, "offset")):
    panel = g[:, ix, :]
    c = 0.6 * np.abs(panel[:, 20:]).max()
    a.imshow(panel.T, extent=[op.angles[0], op.angles[-1], z[-1], 0], cmap="gray", vmin=-c, vmax=c, aspect="auto")
    a.set(title=f"ADCIG at x={x[ix]:.0f} m, {name}", xlabel="reflection angle [deg]")
    a.set_xlim(-60, 60)
fig.tight_layout()
if args.save:
    fig.savefig(args.save, dpi=110)
    print("saved", args.save)
else:
    plt.show()
