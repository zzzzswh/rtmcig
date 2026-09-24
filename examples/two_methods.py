"""Poynting versus offset/slant-stack angle gathers on a single shot over a
horizontal reflector, compared with the angles the geometry predicts.

    python examples/two_methods.py [--save out.png]
"""
import argparse
import math
import time

import numpy as np

from rtmcig import RTMCIG

parser = argparse.ArgumentParser()
parser.add_argument("--save", default=None)
parser.add_argument("--device", default="auto")
args = parser.parse_args()

nx, nz, dx = 201, 61, 10.0
z0 = 295.0                                          # interface between cells 29 and 30
vel = np.full((nx, nz), 2000.0, np.float32)
v_true = vel.copy()
v_true[:, 30:] = 2500.0
xs = 1000.0
srcs = np.array([[xs], [0.0]])
recs = np.stack([np.arange(nx) * dx, np.zeros(nx)])
nt, dt, f0 = 601, 0.002, 20.0
common = dict(nh=36, hmax=90.0, signed=True, halfderiv=True, shot_batch=1, device=args.device)

ops = {
    "poynting": RTMCIG(vel, dx, dx, srcs, recs, nt, dt, f0=f0, method="poynting", **common),
    "offset": RTMCIG(vel, dx, dx, srcs, recs, nt, dt, f0=f0, method="offset", offsets=30, **common),
}
op = ops["poynting"]
data = op.model(v_true, max_vel=2500.0) - op.model(max_vel=2500.0)

results = {}
for name, o in ops.items():
    t0 = time.time()
    results[name] = o.migrate(data)
    print(f"{name}: {time.time() - t0:.1f}s")

angles = op.angles
xs_cig = np.array([-400.0, -300.0, -200.0, -100.0, 100.0, 200.0, 300.0, 400.0]) + xs
print("\nangle of the reflector event (amplitude-weighted around the peak bin):")
print("   x [m]   geometry   poynting   offset")
for xc in xs_cig:
    ix = int(round(xc / dx))
    expected = math.degrees(math.atan((xc - xs) / z0))
    row = []
    for name in ("poynting", "offset"):
        g = np.abs(results[name][:, ix, 26:34]).sum(1)
        k = int(g.argmax())
        sel = np.abs(angles - angles[k]) <= 15.0
        row.append(float((g[sel] * angles[sel]).sum() / g[sel].sum()))
    print(f"  {xc:6.0f}   {expected:8.1f}   {row[0]:8.1f}   {row[1]:7.1f}")

import matplotlib
if args.save:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
image = op.image
clip = 0.5 * np.abs(image[:, 10:]).max()
ax[0].imshow(image.T, extent=[0, (nx - 1) * dx, (nz - 1) * dx, 0], cmap="gray", vmin=-clip, vmax=clip, aspect="auto")
ax[0].plot([xs], [0], "r*")
ax[0].axvline(xs + 300.0, color="c", lw=0.8)
ax[0].set(title="single-shot RTM image", xlabel="x [m]", ylabel="z [m]")
ix = int(round((xs + 300.0) / dx))
expected = math.degrees(math.atan(300.0 / z0))
for a, name in zip(ax[1:], ("poynting", "offset")):
    panel = results[name][:, ix, :]
    c = 0.7 * np.abs(panel[:, 10:]).max()
    a.imshow(panel.T, extent=[angles[0], angles[-1], (nz - 1) * dx, 0], cmap="gray", vmin=-c, vmax=c, aspect="auto")
    a.axvline(expected, color="y", lw=0.8, ls="--")
    a.axhline(z0, color="y", lw=0.6, ls="--")
    a.set(title=f"ADCIG at x={xs + 300:.0f} m, {name}", xlabel="reflection angle [deg]")
fig.tight_layout()
if args.save:
    fig.savefig(args.save, dpi=110)
    print("saved", args.save)
else:
    plt.show()
