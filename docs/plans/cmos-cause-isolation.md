# CMOS discrepancy: controlled cause isolation

Keep the imported material coefficients, source polarization, and physical detector
plane fixed. Run broadband GPU controls sequentially on the local RTX3090. Save
inputs, raw complex fields/fluxes, analytic comparisons and execution metadata.

1. **Resolve the measurement gate.** Record native staggered fields on both sides
   of a planar silicon boundary. Compare current cross-interface interpolation
   with one-sided reconstruction at the same physical plane. Sweep depth spacing,
   fractional interface position, wavelength and incidence. Distinguish a sampling
   defect from a field-update defect before changing production code. Target
   <0.5 percentage-point broadband flux error and refinement convergence.
2. **Check the planar optical stack.** Use transfer-matrix references for the
   imported air/silica/filter/SiN/silicon stack and metal films. Check reflected,
   transmitted and absorbed power plus complex field phase. Hold mesh, source and
   boundary conditions fixed across corrections; compare both absorber and CPML
   if source/reflection feedback remains significant.
3. **Localize patterned-geometry effects.** Add filter quadrants, metal shield and
   interconnects, then lenses. Compare component controls with the full sensor on
   the same grid and reuse the empty reference only when its configuration is
   unchanged. Compare native and finer grids where an effect is sensitive. Exact
   Tidy3D component parity needs numerical reference outputs; do not infer it
   from the published plot or run a paid cloud simulation.
4. **Apply demonstrated fixes and rerun the sensor.** Include focused regression
   tests, run affected CI checks, then execute the broadband sensor/reference pair
   and refresh plots, notebook and PR. Separate measured causes from hypotheses;
   do not refit materials or move detector planes to force agreement.

## Execution record

- **Completed:** normal-incidence broadband native-plane reconstruction and
  production one-sided H sampling, with focused coordinate/derivative-jump tests.
  Existing 550 nm oblique constitutive controls are retained; broadband tilted
  interface reconstruction is not certified by the normal-incidence sweep.
- **Completed:** planar RGB stacks with complex transfer-matrix references and
  depth factors 1/2/4/8, plus factor 16 for blue. The scan/fori-loop monitor clock
  defect was identified and fixed; fine-grid convergence is restored.
- **Completed:** native two-plane incident calibration and its independent
  synthetic-wave tests. An end-to-end reference test also catches input-grid
  versus solver-grid origin mistakes.
- **Completed:** corrected full sensor, no-lens sensor, and patterned-filter
  dielectric stack without metal/lenses, each at 510 fs and 200 frequencies on
  the RTX3090. Their central R/G/B efficiencies are 9.725/20.652/9.203%,
  7.949/14.194/5.831%, and 10.463/19.123/6.912%, respectively.
- **Completed:** refreshed broadband silicon and aluminum controls and planar
  fixed-grid timestep/boundary checks. At 2.5 nm, silicon transmission error is
  below 0.44 percentage points and aluminum-film reflection error is 0.250 points.
- **Completed:** full-sensor 20 nm optical-depth cap at unchanged timestep. It
  preserves all original edges, increases z cells from 430 to 519, and retains
  original lateral, silicon, calibration-region and absorber grids. Central
  R/G/B becomes 9.898/21.365/9.807%, farther from the published plot estimates.
  This rules out this depth correction as a sufficient fix, not spatial error
  more generally. Maximum spectral changes are 0.297/0.807/0.815 points.
- **Completed:** executed full-device notebook with normalized field plots and
  saved complex field arrays (713.74 seconds on RTX3090). Both it and the
  diagnostic analysis notebook executed seven code cells with zero errors.
  Independent full-sensor spectra agree to numerical precision.
- **Validation:** 2,028 selected local tests pass (one skip, one expected failure),
  including calibration/origin, targeted-mesh and field-plot normalization tests. The solver
  fixes are pushed in `878fb34e`; that commit passed hosted CI. Calibration and
  diagnostic runners are in `2f5496e9`, and the targeted-depth setup in `9a011a94`. The field-plot fix is in
  `c67dea02`; all hosted checks for that implementation commit passed.

Original numerical Tidy3D outputs remain unavailable. This is a limit on exact
solver-to-solver attribution, not on the analytical controls. No cloud simulation
has been submitted. Previous results remain under `docs/reviews/cmos-polarized`;
new raw controls and the interpretation are in
[the isolation report](../reviews/cmos-isolation/README.md).

## Remaining attribution work

Neither this depth control nor the earlier 20-to-15 nm lateral control supports
a single resolution change as a sufficient explanation. The demonstrated
defects are fixed, but the full sensor is not spatially
converged. The next controlled comparison requires numerical Tidy3D outputs
with explicit detector normalization, starting with the planar stack and then
adding patterned filters, metals and lenses. Refine lateral spacing separately
from the complete depth grid, retaining physical geometry and recording the
interface policy. A matched operator comparison must distinguish BeamZ's
diagonal causal averaging from Tidy3D's dielectric contour-path/metal-staircase
settings. Do not change material poles or detector positions to fit a plot.

## Follow-up: public-reference and paper reconstruction

The user confirmed no numerical Tidy3D files are available. They are no longer
being treated as a prerequisite: full published curves have been digitized,
filter curves independently cross-checked, and the grid log used to select
between conflicting aluminum inputs. A fixed-grid RTX3090 metal substitution
changes central R/G/B by only +0.030/+0.292/+0.010 points, excluding that
substitution as the explanation of the red/blue gap.

The user-provided Deinega–Valuev (2007) paper identifies a specific limitation:
our diagonal-ε constitutive approximation omits cross-component inverse-tensor
coupling at curved dispersive metal interfaces. The [paper audit](../reviews/cmos-reverse-engineering/paper-interface-audit.md)
records the code comparison, local numerical examples, Tidy3D policy differences
and proposed curved-metal validation gates. It does not claim the CMOS gap is
already attributed or that changing one diagonal formula implements the full tensor.

## Material-dependent default policy

Completed in `2632030e`: optical metals use point-sampled staircasing, while
dielectrics retain causal diagonal Farjadpour averaging. This preserves the
documented diagonal constitutive law and makes the metal exception explicit.
The 20 nm RTX3090 control gives R/G/B **9.906/20.821/9.223%**, changing the previous
all-diagonal values by **+0.181/+0.169/+0.020 percentage points**. Raw comparisons and
interpretation are in the [isolation report](../reviews/cmos-isolation/README.md).
The affected Python suite passes 829 tests, including 20 policy regressions;
53 Rust tests pass (one ignored). Full dispersive tensor smoothing and curved
metal Mie convergence remain separate work.


## Full-sensor outer z boundary control

Changing only the outer z absorption changes central R/G/B by **−0.017/−0.009/+0.0003 percentage points**; maximum changes across the full spectra are **0.058/0.053/0.024 points**. These are much smaller than the remaining ~3.72/~2.98-point red/blue gaps. The full-sensor control therefore does not support the outer z absorbers as the dominant cause on this grid. All saved arrays are finite; the largest 300–510 fs change is 0.0354 points, so the smallest boundary differences should not be interpreted as precision measurements.

This leaves the discrepancy unresolved. Periodic seams, internal dielectric-interface operators, full spatial convergence and fidelity to the published reference remain separate attribution questions. Demonstrated planar discretization errors do not by themselves identify which of these explains the sensor mismatch.

The controlled RTX3090 run takes 566.31 seconds after setup on the same 150×150×430 grid, with unchanged timestep, geometry, material policy and detector reconstruction. R/G/B efficiencies with CPML are **9.889/20.812/9.224%**, versus absorber **9.906/20.821/9.223%**. Each uses its own native-wave incident calibration. Tidy3D itself uses adiabatic z absorbers, not CPML; this is a sensitivity test, not a replacement reference. Raw arrays, boundary specifications, source hashes, validation and the comparison plot are under `docs/reviews/cmos-isolation/components/full-staircase-metal-cpml`. The analysis notebook executes ten code cells without errors.
