# Causal polarized dispersion: local validation

The default diagonal Farjadpour policy now applies to scalar dispersive media.
The complete causal normal response is harmonic and the tangential response is
arithmetic. The implementation retains constituent polarization histories at
native laminar interface supports; it does not harmonically average residues.
See the [update derivation](../../plans/dispersive-diagonal-update.md).

## Measured baseline

The local RTX3090 completed a 25 nm, 510 fs device/reference pair with 200
frequencies, exact imported material coefficients, and unchanged detector planes.
Fields and auxiliary polarization remained finite. Device execution, including
JIT compilation, took 214.5 seconds; the reference took 93.9 seconds. These are
local run times, not isolated performance benchmarks.

| Treatment | Red, 650 nm | Green, 550 nm | Blue, 450 nm |
| --- | ---: | ---: | ---: |
| Previous volume baseline | 9.565% | 19.642% | 7.888% |
| Causal diagonal, 25 nm | 9.616% | 19.740% | 7.909% |
| Causal diagonal, graded 20/5 nm | 10.109% | 22.002% | 9.941% |
| Reference z, 20 nm xy, external absorbers | 9.785% | 20.615% | 8.849% |
| Published Tidy3D plot, approximate | 6% | 20% | 6% |

The maximum spectral changes from the previous baseline are 0.057, 0.182 and
0.056 percentage points for R/G/B. This change alone does **not** close the
reference disagreement. The historical uniform volume baseline also reconstructed
its instantaneous epsilon from cell averages; that inconsistency is now fixed,
so the historical comparison is not a pure constitutive-policy experiment.

![Calibrated spectra](comparison.png)

Files in `25nm/` retain the raw device/reference spectra, calibrated efficiencies
and execution metadata. Wavelength arrays can be descending; sort before
interpolating central values. Green efficiency sums both green apertures, and all
channels use the same full-cell empty-reference incident power.

## Independent validation

Constitutive tests cover normal/tangential weights, fractions 0 and 1, identical
materials, static limits, shared epsilon-infinity, Lorentz and Drude responses,
and dispersive/dispersive mixtures. The expected response uses the trapezoidal
frequency mapping. Geometry tests cover equal-epsilon material identity;
continuation, periodic seams, memory estimates and immutable inputs are checked.

A 30-degree periodic oblique half-space fixture uses a transverse period of one
550 nm incident transverse wavelength, so no nonzero Bloch phase is required.
GPU results at 5 nm and interface shifts of 0, 0.25 and 0.5 cells give
maximum reflection/transmission errors of 0.349/0.398 percentage points for the
polarized silicon TE/TM cases. At 10 nm these maxima are 1.420/1.451 percentage
points: refinement matters, and the coarse cases do not meet the 0.5-point gate.
TE agrees with explicit volume averaging. At the quarter-cell shift, silicon TM
transmission error decreases from 0.318 to 0.239 percentage points. Improvements
are not universal: half-cell TM reflection is slightly worse (0.349 versus 0.321).

For aluminum at 5 nm and a quarter-cell shift, TM reflection error decreases from
0.305 to 0.149 percentage points, and transmission error is 0.021 percentage
points. These are single-frequency, 550 nm controls, not broadband certification.
Raw cases are retained in `oblique/`. All runs used the local RTX3090 and CPML.

The ordinary absorber produced substantial feedback in that oblique fixture;
CPML is used to isolate its constitutive error. This is a boundary-control finding,
not evidence that changing interface averaging alone fixes a sensor simulation.

## Detector-plane and timestep controls

Broadband normal-incidence silicon controls retain the detector at the physical
interface. Maximum transmission error is 3.072 percentage points at 5 nm and
1.691 at 2.5 nm. At three cells inside silicon, comparison with the correctly
attenuated analytical flux gives 0.156 and 0.085 percentage points. Halving the
Courant factor changes the measured spectra by less than 0.040 percentage points
across all checked planes; doubling the absorber changes interface transmission
by 0.013 percentage points or less. Raw spectra and reports are in `interfaces/`
and `interfaces-half-dt/`.

The interface-plane accuracy gate is **not passed**. This identifies a separate
flux-colocation error; the sensor detector has not been moved to hide it.
The 20/5 nm sensor's 510 fs run is temporally settled to below 0.06 percentage
points against its earlier checkpoint, so simply running longer is also unlikely
to explain the multi-point red/blue discrepancy.

## Newly audited reference grid

The public snapshot's exact Tidy3D version, 2.12.0.dev3, reconstructs a
390 × 390 × 430 grid locally. No cloud job is submitted. Its 40 absorber cells
extend **outside** the specified physical domain: 2.19549 µm above and
0.465116 µm below it. The previous BeamZ setup used 400 nm absorbers inside the
physical box. The boundary position and thickness were therefore not matched.

`examples/data/cmos_rgb/tidy3d_reconstructed_grid.npz` stores the reconstructed
edges; the neighboring JSON records the version and snapshot hash. This is a
local reconstruction, not an exported server-result mesh. The control in
`examples/cmos_reference_mesh.py` retains its z edges and physical geometry,
including silicon through the lower absorber. The reference control also retains
the reconstructed timestep, source pulse offset, and cubic absorber strength
converted from Tidy3D's documented `2*epsilon0/dt` units to SI conductivity. Lateral spacing remains explicit
to fit the RTX3090. It is not a claim of common-mesh agreement.

The complete reference-depth notebook executed all six code cells on the RTX3090
with no errors in 967.6 seconds. Device execution including compilation took
659.3 seconds; all fields and polarization histories remained finite. Its actual
300-to-510 fs changes are 0.0017/0.0140/0.0011 percentage points for R/G/B.
Spectra, grid, absorber parameters and field plots are retained in
`reference-z-20nm-absorber/`; the executed notebook embeds the comparison.

Green and blue improve relative to the graded run, but red and blue still do not
closely match the published plot. This changes several inputs together, so it
does not isolate the contribution of boundary placement. The full-sensor
2.5 nm depth sweep and matched component comparisons remain open; the failed
interface-plane gate should be addressed before another expensive sweep.

## Limits

Local validation passed 1,996 tests, with one skip and one expected failure.
Changed-line coverage is 94%; all configured coverage floors pass.

Original numerical Tidy3D output spectra remain unavailable. Reference output
values above are visual estimates from the [published notebook](https://www.flexcompute.com/tidy3d/examples/notebooks/CMOSRGBSensor/).
Its dielectric and metal interface policies differ (`PolarizedAveraging` and
`Staircasing`). Tidy3D documents the former as a contour-path approximation,
which is not the same constitutive construction as our diagonal Farjadpour law. The diagonal method omits off-diagonal tensor coupling and retains
native volume fallback at ambiguous normals. Monitor colocation at a material
jump, depth/time convergence, and sensor boundary sensitivity remain separate
accuracy gates; finite fields alone do not establish agreement.
