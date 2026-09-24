"""Timing of rtmcig on the current device.

    python benchmarks/bench.py                          # default: 401 x 201 grid, 8 shots, nt = 1500
    python benchmarks/bench.py --nx 801 --nz 401 --ns 16 --nt 3000 --method offset --image_dt 2

Prints the wall time per shot for the forward modelling of the data and for the
migration, the snapshot memory per shot and the peak device memory. Use --shot_batch
to fix the batch size instead of the automatic one.
"""
import argparse
import time

import numpy as np
import torch

from rtmcig import RTMCIG
from rtmcig._utils import human_bytes

parser = argparse.ArgumentParser()
parser.add_argument("--nx", type=int, default=401)
parser.add_argument("--nz", type=int, default=201)
parser.add_argument("--dx", type=float, default=10.0)
parser.add_argument("--ns", type=int, default=8)
parser.add_argument("--nr", type=int, default=401)
parser.add_argument("--nt", type=int, default=1500)
parser.add_argument("--dt", type=float, default=0.002)
parser.add_argument("--f0", type=float, default=15.0)
parser.add_argument("--nh", type=int, default=32)
parser.add_argument("--method", default="poynting", choices=["poynting", "offset"])
parser.add_argument("--equation", default="scalar", choices=["scalar", "acoustic"])
parser.add_argument("--offsets", type=int, default=20)
parser.add_argument("--image_dt", type=int, default=1, help="imaging interval in samples")
parser.add_argument("--store", default="device", choices=["device", "cpu"])
parser.add_argument("--store_dtype", default="float16", choices=["float16", "bfloat16", "float32"])
parser.add_argument("--shot_batch", default="auto")
parser.add_argument("--aperture", type=float, default=None)
parser.add_argument("--device", default="auto")
parser.add_argument("--repeat", type=int, default=1)
args = parser.parse_args()

vel = np.full((args.nx, args.nz), 2000.0, np.float32)
vel[:, args.nz // 2:] = 2500.0
v_mig = np.full((args.nx, args.nz), 2000.0, np.float32)
xmax = (args.nx - 1) * args.dx
srcs = np.stack([np.linspace(0.2 * xmax, 0.8 * xmax, args.ns), np.zeros(args.ns)])
recs = np.stack([np.linspace(0.0, xmax, args.nr), np.zeros(args.nr)])
shot_batch = args.shot_batch if args.shot_batch == "auto" else int(args.shot_batch)

op = RTMCIG(v_mig, args.dx, args.dx, srcs, recs, args.nt, args.dt, f0=args.f0, nh=args.nh, hmax=60.0,
            method=args.method, equation=args.equation, offsets=args.offsets,
            image_dt=args.image_dt * args.dt, store=args.store, store_dtype=args.store_dtype,
            shot_batch=shot_batch, aperture=args.aperture, device=args.device, verbose=True)
dev = op.device
print(op)
print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(dev)})" if dev.type == "cuda" else
                          f" ({torch.get_num_threads()} threads)"))
print(f"grid {args.nx} x {args.nz}, {args.ns} shots x {args.nr} receivers x {args.nt} samples, "
      f"{op.n_snapshots} imaging steps, snapshots {human_bytes(op.snapshot_bytes(1))} per shot")


def timed(fn):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    out = fn()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return out, time.perf_counter() - t0


data, t_model = timed(lambda: op.model(vel, max_vel=2500.0))
print(f"modelling: {t_model:.2f}s total, {1e3 * t_model / args.ns:.1f} ms per shot")

for r in range(args.repeat):
    adcig, t_mig = timed(lambda: op.migrate(data))
    peak = f", peak device memory {human_bytes(torch.cuda.max_memory_allocated(dev))}" if dev.type == "cuda" else ""
    print(f"migration ({args.method}, {args.equation}): {t_mig:.2f}s total, "
          f"{1e3 * t_mig / args.ns:.1f} ms per shot{peak}")
print("adcig", adcig.shape, "finite:", bool(np.isfinite(adcig).all()))
