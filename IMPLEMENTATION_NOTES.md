# Implementation notes

Design decisions, the reasoning behind them and what was verified numerically.
Companion to the README; read that first.

## 1. Why Deepwave, and how it is driven

Deepwave (>= 0.0.27) provides propagators for the constant-density scalar wave
equation (`deepwave.scalar`) and the variable-density velocity-pressure system on a
staggered grid (`deepwave.acoustic`), both with C-PML absorbing boundaries, 2nd-order
time stepping, 2/4/6/8th-order space stencils, automatic CFL sub-stepping (sources are
FFT-upsampled, receivers downsampled) and prebuilt CPU/CUDA kernels. Writing another
staggered-grid + PML propagator would have duplicated exactly this.

What an RTM needs on top of a propagator is access to the wavefield at every imaging
step, without leaving the compiled time loop. Deepwave's `forward_callback` does that:
it is called before each block of `callback_frequency` user steps with a
`CallbackState` whose `get_wavefield(name, view="inner")` returns a `(B, nz, nx)`
view of the current field. `rtmcig` never uses Deepwave's autograd; RTM is a
cross-correlation of two forward propagations, and the memory-hungry checkpointing
of a differentiable propagator is not needed.

Facts about the callback that the code relies on (checked against the 0.0.27 source):

- `state.step` is the *user* step, `state.dt` the *inner* step (`dt / step_ratio`).
- The callback fires at user steps `0, cf, 2cf, ... < nt`. At step `s` the field
  `wavefield_0` is `u(s)` (the value that receivers at step `s` record) and
  `wavefield_m1` is one inner step earlier, so `(u - u_m1) / state.dt` is `u_t`.
- Views alias Deepwave's ping-pong buffers and are overwritten by the next block;
  everything is copied or consumed inside the callback.
- Source cells and receiver cells must be unique within a shot; `IGNORE_LOCATION`
  (`-2^31`) marks padding. Receivers falling in one cell are summed by `rtmcig`.
- The scalar engine adds `-v^2 dt^2 f(t)` to `u(t+1)`; the acoustic engine adds
  `dt K s(t)` to `p(t+1)` with `K = rho v^2` at the source cell.

## 2. Time pairing: the reversed run is Deepwave's exact adjoint

Back-propagation is the same propagator run on the time-reversed traces. A dot-product
test (`<F f, d> = <f, F^T d>`) against Deepwave's forward modelling passes to `8e-8`
relative error when the field recorded at reversed step `m` is paired with physical
step `nt - 1 - m`. So, with imaging interval `L`, the traces are zero-padded to
`nt_prop = K L + 1` (`K = ceil((nt-1)/L)`), source snapshots are stored at steps
`L, 2L, ..., (K-1)L` (snapshot `k = s/L - 1`) and the receiver callback at reversed
step `m = k' L` uses snapshot `k = K - k' - 1`. Nothing is stored for step 0 (empty
field) or step `K L` (pairs with the empty receiver field).

## 3. The 2D wavelet and `halfderiv`

The plain RTM image of a reflector in 2D is not the zero-phase autocorrelation of the
wavelet: a specular reflection is **one** cylindrical Green's function (the mirror
source), not the product of two. The source wavefield carries a 45-degree phase and
`|w|^-1/2` from its own Green's function; the data carry the same once; the
back-propagation multiplies by the conjugate Green's function, cancelling the data's
factor but leaving the source wavefield's. The cross-correlation therefore has a net
45-degree rotation and a low-frequency tilt. This was verified with a single
zero-offset pair: the imaged wavelet is mixed-phase; after applying
`conj(sqrt(1 - rho e^{-iw}))` (the adjoint half-derivative of Madagascar's
`sf_halfint`, the same filter `kirchcig` uses) to the data it becomes symmetric and
centred on the interface to within half a cell. `halfderiv=False` by default, as in
`kirchcig`; the raw wavelet then has its largest lobe on the interface and a strong
trailing lobe.

That accounts for one Green's function, not for the acquisition. With a receiver line
the receiver wavefield at an image point is `sum_r conj(G(x, r)) D(r)`, and for a dense
line that sum is a stationary-phase integral along the receivers, which contributes
another `e^{i pi/4} |w|^-1/2`. So with a receiver spread the image is rotated by about
90 degrees without the filter and about 45 with it, and it takes a full time derivative
(`halfderiv` plus one more `half_derivative` of the data) to make it zero-phase. Measured
in the test model (reflector at 195 m, averaged over x = 380-420 m, phase of the analytic
signal at the envelope peak): 17 shots over 400 m into all 81 receivers give +90 degrees
without the filter, +48 with `halfderiv` and +6 with the full derivative. A narrow shot
aperture compared with the depth makes the residual larger and less regular (9 shots over
200 m: +78 with `halfderiv`), because the image point is then not surrounded by
contributing shots. `halfderiv` keeps its single-trace definition, `kirchcig`'s
convention; `test_halfderiv_with_receiver_spread` checks the 17-shot numbers.

Polarity: with `halfderiv=True` the zero-offset image of a velocity increase is a
negative symmetric lobe in Deepwave's sign convention (its source term is
`-v^2 f`), so the image is `-sum_t u_s u_r`, which makes a velocity (impedance)
increase image as a positive peak for data modelled with `op.model` or with Deepwave.

## 4. The two engines give the same answer

Injecting `-(1/rho(x)) int a dt` (trapezoidal) into the acoustic engine makes its
pressure the scalar engine's wavefield: direct waves correlate to 0.99, reflection
data to 0.998, images to 0.99 and gathers to ~0.95-0.99 between the engines
(remaining differences: the half-step time staggering and the different stencils).
Without this the acoustic image would be the negative second time derivative of the
scalar one and the two engines would disagree in polarity, phase and by a factor
`K ~ 1e9-1e10` in scale.

A subtlety found on the way: Deepwave's acoustic engine records, at a cell that is
both a source and a receiver, a direct-wave sample that depends on the internal
step ratio (`max_vel`). Two data sets that will be subtracted (true minus background,
to remove the direct wave) must therefore be modelled with the same `max_vel`;
`op.model(..., max_vel=...)` exposes it and `demo_data()` uses it.

## 5. Poynting angles

`P = -u_t grad(u)` (scalar) or `P = p v` (acoustic), box-smoothed over
`poynting_smooth` cells (default a quarter of the dominant wavelength) so that the
direction is energy-weighted. The source direction is stored as an int16 angle. The
signed reflection angle is `theta = 0.5 atan2(n_s x n_r, n_s . n_r)`: no
normalisation, no acos of a rounded dot product, and positive when the source is on
the -x side of the receiver. Contributions are split linearly between the two bins
whose centres bracket `theta` (edge half-bins clamp), so `adcig.sum(0)` equals the
image to float precision when the axis covers all angles.

Accuracy, measured on a horizontal reflector under a single shot: the source-side
direction is within ~1 degree of geometry. The receiver-side direction is exact for a
single receiver, but the summed receiver wavefield at the reflection point, at the
reflection time, is the coherent superposition of the back-propagated arrivals of every
receiver whose isochron passes within a wavelet of the point. For a flat reflector at
depth `z` that is the whole Fresnel zone, `~sqrt(lambda z / 2)` wide, and those arrivals
come from directions spanning tens of degrees on either side of the specular one; the
Poynting vector of the sum is their energy-weighted mean, and because the far-side
receivers span a larger range of angles than the near-side ones the mean lands outside
the specular direction. The bias therefore grows with the receiver spread and scales
like `sqrt(lambda / z)`: 7-10 degrees at 22-34 degrees for a reflector at 200-300 m
under a 0.8-2 km spread (the test models), a few degrees at depths of a kilometre or
more. Binning each image point at the angle of its strongest correlation sample instead
of instant by instant was tried and changes nothing (the peak sample itself is biased),
so the instantaneous binning was kept. `method="offset"` does not have this problem
(1-2 degrees in the same tests) because the offset gathers measure the local slope of
the correlation, not a direction of the summed field. Multi-shot stacking reduces the
crosstalk noise that single-shot gathers show at other angles.

## 6. Offset gathers and the slant stack

`I(x, z, h) = sum_t u_s(x-h) u_r(x+h)` over the offsets `-offsets*step ... +offsets*step`
cells, accumulated with shifted slices (index_select for `cig_x` subsets). The angle
transform is `A(z, theta) = sum_h w_h I(z + h tan theta, h) / sum_h w_h`, evaluated by
multiplying the `z`-FFT of each offset panel with `exp(i k_z h tan theta)` (exact
sub-sample shifts), with zero padding in depth so that nothing wraps, a cosine taper
`w_h` on the outer `offset_taper` of the offset axis and chunking over columns.
Dividing by the taper weight rather than by `nho` keeps a focused event at the
image's amplitude (with the default taper, `1/nho` left it about 10% low). The sign of
the tangent is chosen so that positive angles again mean "source on the -x side";
the tests check this against geometry and against the Poynting result. Right below a
shot the single-shot ODCIG is a vertical strip over all offsets (the source and
receiver wavefields are symmetric there); away from it, and after stacking shots, the
gathers focus at `h = 0`.

## 7. Memory

Poynting snapshots cost 4 bytes per cell per imaging step (float16 field scaled per
snapshot and shot + int16 direction); offset snapshots 2 bytes (float16) or 4
(float32). `image_dt = k dt` divides that by `k`; the cross-correlation only needs
the Nyquist rate of the data. `shot_batch="auto"` fits `memory_fraction` of the free
device memory and falls back to pinned host storage when one shot does not fit.
The host store keeps two device buffers and streams the next snapshot on a side
stream while the current one is imaged (`record_stream` protects the temporaries;
`wait_stream` orders reuse of a buffer after the kernels that read it). On the CPU the
"device" store is host memory anyway and Deepwave parallelises over the shots of a
batch with OpenMP; `shot_batch="auto"` takes one shot per thread (up to 8) but no more
than fit into `memory_fraction` of the available host memory (`MemAvailable` on Linux),
and warns when a single shot does not fit, since there is nothing to fall back to.

`aperture` propagates each batch on the columns between its sources and live
receivers plus the aperture, replicating the model edges into the PML; the batch
accumulators are added into the global image/gather arrays through flat cell index
tables, which is also how `cig_x` column subsets and angle bins are addressed
(`index_add_` on a flat float32 buffer).

## 8. Geometry

Positions in metres, `(2, n)` rows `(x, z)` as in `kirchcig`; receivers `(2, nr)` or
`(ns, 2, nr)`; `NaN` marks a dead trace (zero amplitude, `IGNORE_LOCATION`). Nearest
cells by default. `interpolation="hicks"` uses Deepwave's Kaiser-windowed sinc
interpolation (Hicks 2002) per shot for the live receivers; the model is replicate-
padded by the half-width so that windows near the edges stay on the grid (not at the
top with `free_surface`, where Hicks's own free-surface images are used).

## 9. What is not here (yet)

- 3D. Deepwave propagates 3D; the imaging code is written for `(nz, nx)` fields.
- Elastic RTM (Deepwave has an elastic propagator; P/S separation would be needed).
- Demigration / a least-squares operator pair. RTM is the adjoint of Born modelling,
  which Deepwave can compute through autograd; wiring the two into a linear operator
  is a natural extension.
- Dip-guided angles (source direction + reflector normal from the image), which would
  remove the receiver-side Poynting bias for continuous reflectors.
