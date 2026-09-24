"""The built-in demo, plotted: two-layer model, 5 shots, 51 receivers.

    python examples/plot_adcig.py [--save out.png]
"""
import argparse
import time

import numpy as np

from rtmcig import RTMCIG

parser = argparse.ArgumentParser()
parser.add_argument("--save", default=None)
parser.add_argument("--method", default="poynting", choices=["poynting", "offset"])
parser.add_argument("--device", default="auto")
args = parser.parse_args()

op = RTMCIG.demo(method=args.method, signed=True, nh=36, halfderiv=True, device=args.device, verbose=True)
print(op)
t0 = time.time()
data = op.demo_data()
print(f"demo data {data.shape} in {time.time() - t0:.1f}s")
t0 = time.time()
adcig = op.migrate(data)
print(f"migration in {time.time() - t0:.1f}s; adcig {adcig.shape}, stack == image: "
      f"{np.abs(adcig.sum(0) - op.image).max() / np.abs(op.image).max():.1e}")

import matplotlib
if args.save:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

x, z = op.x, op.z
ix = op.nx // 2
image = op.image
clip = 0.5 * np.abs(image[:, 10:]).max()
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1.4, 1.4, 1]))
ax[0].imshow(data[len(data) // 2].T, aspect="auto", cmap="gray",
             extent=[op.survey.recs[0, 0, 0], op.survey.recs[0, 0, -1], (op.nt - 1) * op.dt, 0],
             vmin=-0.3 * np.abs(data).max(), vmax=0.3 * np.abs(data).max())
ax[0].set(title="middle shot record", xlabel="receiver x [m]", ylabel="t [s]")
ax[1].imshow(image.T, extent=[x[0], x[-1], z[-1], z[0]], cmap="gray", vmin=-clip, vmax=clip, aspect="auto")
ax[1].axhline(op._demo["z_reflector"], color="y", lw=0.6, ls="--")
ax[1].axvline(x[ix], color="c", lw=0.8)
ax[1].plot(op.survey.srcs[0], op.survey.srcs[1], "r*")
ax[1].set(title="RTM image", xlabel="x [m]", ylabel="z [m]")
panel = adcig[:, ix, :]
c = 0.7 * np.abs(panel[:, 10:]).max()
ax[2].imshow(panel.T, extent=[op.angles[0], op.angles[-1], z[-1], z[0]], cmap="gray", vmin=-c, vmax=c, aspect="auto")
ax[2].axhline(op._demo["z_reflector"], color="y", lw=0.6, ls="--")
ax[2].set(title=f"ADCIG at x={x[ix]:.0f} m ({args.method})", xlabel="reflection angle [deg]")
fig.tight_layout()
if args.save:
    fig.savefig(args.save, dpi=110)
    print("saved", args.save)
else:
    plt.show()
