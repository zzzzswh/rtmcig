# Changelog

## 0.1.0

First release.

- `RTMCIG` operator and one-shot `migrate`: prestack data `(ns, nr, nt)` to angle-domain
  common-image gathers `(nh, nx, nz)` by reverse-time migration.
- Wave propagation through Deepwave's `scalar` (constant-density) and `acoustic`
  (variable-density velocity-pressure, staggered grid) engines with C-PML boundaries;
  the wavefields are read through Deepwave's `forward_callback`, so the whole time loop
  stays in compiled C/CUDA. The acoustic engine injects `-(1/rho) int f dt` so that it
  reproduces the scalar engine's data, phase, polarity and scale.
- Angle decomposition by Poynting vectors (`method="poynting"`, signed or unsigned
  angles, linear bin splitting so the gathers sum exactly to the image) or by the
  subsurface-offset extended imaging condition and a Fourier-domain slant stack
  (`method="offset"`, `op.odcig` kept).
- Exact adjoint time pairing of the reversed-time receiver run, `halfderiv` 2D phase
  correction (`kirchcig` convention), polarity: a velocity increase images positive.
- Source-wavefield snapshots in float16 (per-snapshot scaled), bfloat16 or float32, on
  the device or in pinned host memory streamed on a side CUDA stream; `image_dt` to
  subsample the imaging steps; automatic shot batching against the free device memory.
- Geometry: metres in, nearest cell or Hicks sub-cell interpolation, dead traces
  (`NaN`), receivers sharing a cell merged, per-shot receiver spreads, lateral
  `aperture` windows, `cig_x` output columns.
- Options: `illumination` compensation, `laplacian` filter, `free_surface`, `accuracy`,
  `pml_width`, per-shot wavelets, float64 propagation.
- Forward modelling with the same geometry (`op.model`), demo, tests, examples, benchmark.
