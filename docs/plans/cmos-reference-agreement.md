# Resolving CMOS numerical disagreement

Status: investigation plan, not a claim of agreement. No additional full-sensor
GPU runs are required to repair CI; the experiments below are the next accuracy
work, to be run sequentially on the local RTX3090.

## What the evidence currently supports

- Exact RGB and aluminum material imports corrected known input errors. Analytic
  permittivity evaluation now matches the saved reference coefficients; that
  does not prove identical time-discretized or interface response.
- At 650/550/450 nm the 20 nm lateral / 5 nm depth-refined run gives
  10.10/21.96/9.92 percent; the 15 nm lateral run gives 10.15/21.98/9.92 percent.
  Maximum spectral changes are 0.11/0.28/0.05 percentage points for R/G/B.
  This weakens the hypothesis that lateral resolution alone explains the gap.
- Published reference values near 6/20/6 percent are visual estimates. Original
  numerical flux arrays and the realized reference mesh are missing, so neither
  an exact error curve nor common-mesh agreement has been established.
- A monitor exactly on an air/aSi interface has maximum transmission errors of
  3.07 percentage points at 5 nm and 1.69 at 2.5 nm. Three cells inside silicon,
  comparison with the analytically attenuated flux reduces errors to 0.156 and
  0.085 percentage points. This is a specific unresolved measurement/interface
  discretization issue, not evidence that the full sensor needs only finer xy.
- The saved Tidy3D simulation specifies dielectric `PolarizedAveraging` and metal
  `Staircasing`. BeamZ currently volume-averages dispersive material occupancy
  and residues. These are different interface discretizations. Determine which
  reference policy applies to each pole-residue medium before attributing an
  error to a particular policy. Its `SurfaceImpedance` entry for the separate
  lossy-metal category does not prove that it is used for this aluminum model.
- The graded sensor runs use 300 fs and 31 frequency samples, versus about
  509.4 fs and 200 samples in the reference. Frequency sample count mainly
  affects displayed spectral sampling; runtime and source normalization must
  be controlled independently. The existing 25 nm notebook uses 510 fs and 200
  samples, but also changes the grid and source implementation.
- The reference uses 40 absorber layers; BeamZ uses a physical thickness.
  These are not automatically equivalent on graded grids. Existing absorber
  sensitivity checks cover a planar interface, not the complete sensor.

Sources: [saved validation report](../reviews/cmos-reference-parity/README.md),
[reference notebook](https://www.flexcompute.com/tidy3d/examples/notebooks/CMOSRGBSensor/),
and the checked-in `examples/data/cmos_rgb/tidy3d_reference_simulation.json`.
The snapshot's subpixel policies are inspectable in its `subpixel` field.

## Ordered experiments and decision gates

1. **Freeze and audit the comparison inputs.** Export reference numerical fluxes,
   realized grid, solver/version, stop time, and monitor colocation settings if
   accessible. Record hashes linking the outputs to the material snapshot. Audit
   geometry bounds/overlap order, Ex polarization, source spectrum, all monitor
   positions and areas, both green pixels, and the incident-power denominator.
   Until reference arrays are available, report only approximate plot comparisons
   and independent analytical error. Do not refit materials to force efficiency
   curves to agree.

2. **Close the planar interface test before another expensive sweep.** Use the
   imported silica/SiN/aSi and aluminum models, with TE and TM incidence including
   oblique incidence to test normal electric response. Compare reflection,
   transmission, absorption, and complex fields against transfer matrices on
   uniform and graded grids at 10/5/2.5 nm; add 1.25 nm only where needed.
   Sweep the interface's fractional cell position. Sample flux at the interface
   and at known offsets, comparing every offset to its correct attenuated value.
   A proposed acceptance target is <0.5 percentage-point flux error over the band
   and decreasing error under refinement. Fix discontinuity-aware E/H colocation
   if the error is primarily observational. Keep the sensor detector plane fixed.

3. **Isolate the dispersive interface model.** At the same grid, compare the
   present volume average against reference-matched staircasing where applicable.
   Add polarized/conformal dispersive averaging only if the planar and curved
   geometry controls show it is necessary. It must enforce normal-D and
   tangential-E continuity with a causal, passive time-domain update; reusing
   static harmonic averaging of scalar residues is not sufficient. Require stable
   long-time evolution, correct TE/TM Fresnel response, and positive absorption
   before applying a new interface model to the sensor.

4. **Separate time and depth errors in the sensor.** Hold lateral spacing at
   20 nm, fix 200 frequencies, the materials, detector plane, source path, and
   absorber. First run a 510 fs baseline with intermediate 300 fs outputs; extend
   to 750 fs only if the late-time spectral change exceeds 0.2 percentage points.
   Then refine only metal/AR/silicon-interface depth from 5 to 2.5 nm, using the
   same physical boundaries and monitor coordinates. Repeat the empty reference
   only when the grid/timestep/source or normalization window changes. Proposed
   spatial target: <0.5 percentage-point change throughout the band, including
   steep filter edges. If needed, refine smaller depth regions to 1.25 nm after
   estimating memory. The previous lateral pair is evidence, not a universal
   convergence certificate.

5. **Check time discretization and boundaries independently.** On the selected
   mesh, halve the Courant factor for a short planar control before considering
   a full-sensor repeat; this separates ADE temporal dispersion from mesh error.
   Run a thicker sensor absorber with the optical geometry unchanged and enough
   buffer to keep the source/monitors outside it. Require <0.2 percentage-point
   spectral changes; otherwise revise boundary placement or absorption. Check
   incident, reflected, transmitted, and absorbed power in simplified controls
   to catch sign, area, or normalization mistakes.

6. **Localize any residual gap with component controls.** Remove lenses, then
   interconnect/shield features in matched paired models to identify where the
   disagreement enters. Reuse the incident reference when its configuration is
   unchanged. Prefer one informative control at a time to a parameter sweep.
   Compare the final full sensor to numerical reference spectra and field cuts,
   with a documented error budget for time, depth, lateral grid, interface policy,
   and absorber. Agreement is not achieved merely by finite fields or a smoother
   curve.

## RTX3090 execution budget

Use broadband pulses for every case; no per-wavelength simulations. Reuse existing
results and begin with inexpensive planar controls. After those pass, start with
one 510 fs baseline and one depth-refined device/reference pair; add boundary or
component controls only when the preceding result justifies them. Run large jobs
sequentially, estimate fields plus ADE/DFT memory before launching, and target
under 20 GiB of allocated GPU memory. Save actual grids, timesteps, material
hashes, raw fluxes, stop-time diagnostics, and calibrated spectra for every case.
The numerical thresholds above are proposed accuracy gates, not measured results.
