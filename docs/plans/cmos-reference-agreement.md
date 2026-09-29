# Resolving CMOS numerical disagreement

Status: implementation and investigation plan, not a claim of agreement. No additional full-sensor
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

3. **Implement causal diagonal Farjadpour averaging for dispersive interfaces.**
   This is the intended extension of the existing default, not merely an optional
   later experiment. Follow the implementation specification below. Keep volume
   averaging as an explicit comparison policy and preserve the native rasterizer's
   documented fallback for ambiguous interface normals. Establish analytical
   accuracy and stability before switching the CMOS validation runs to the new
   default. Compare reference-matched metal staircasing separately where applicable;
   a general polarized scheme is not automatically identical to Tidy3D's metal
   policy. Do not change material coefficients to compensate for interface errors.

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

## Dispersive diagonal Farjadpour implementation specification

The present bypasses are in `beamz/design/raster/integration.py` and
`beamz/simulation/api.py`: both force dispersive scenes to `volume`.
`MaterialGrid` also rejects dispersive coefficients with any other policy.
Removing these guards alone would smooth epsilon-infinity while leaving the
polarization response volume-averaged, which is not the intended constitutive law.

For an interface between locally isotropic materials, let f_j denote physical
volume fractions and n_i the interface unit normal's component along E_i.
With s the Laplace frequency, use the complete causal material responses:

```text
A(s) = sum_j f_j epsilon_j(s)
H(s) = 1 / sum_j [f_j / epsilon_j(s)]
epsilon_eff,ii(s) = (1 - n_i^2) A(s) + n_i^2 H(s)
```

This reproduces the scalar-material diagonal projection of the existing native
Farjadpour tensor: arithmetic tangential response and harmonic normal response.
It deliberately omits off-diagonal coupling, as the existing diagonal method does.
It is not a promise of the full tensor method's accuracy for tilted interfaces.
The harmonic response can introduce poles beyond those of either constituent;
rescaling the original residues alone cannot represent it.

### Geometry and material data

Expose native geometry integration's material fractions, interface normal, and
laminar/ambiguous classification on each electric Yee support. Preserve painter
order and reuse the actual integration, rather than estimating normals from
sampled permittivity (different materials can share the same epsilon-infinity).
Store interface data sparsely; retain the existing bulk-material ADE path.
Use physical support volumes on rectilinear grids and treat periodic half-supports
consistently. Include interface geometry, policy, and constituent models in cache
identity. Explicitly reject unsupported intrinsic tensor combinations initially.

### Causal update

Prefer a local constituent-field state realization of H(s), avoiding polynomial
root finding or a fitted effective material per mixed cell. For the normal branch,
constituents share D while E = sum_j f_j E_j. Each constituent retains its own
polarization q_j and satisfies D/epsilon0 = epsilon_inf,j E_j + P_j/epsilon0.
The tangential branch uses a shared E and the volume-weighted polarizations.
Blend the two branch responses with n_i^2 to obtain the diagonal component.

Derive the coupled trapezoidal E/polarization solve from these constraints before
implementation. For a constituent with q' = a q + c E, the step has the form
q_new = A q_old + B(E_old + E_new), with A and B determined by the timestep.
Eliminate the local constituent fields algebraically in the normal branch, solve
for the new macroscopic E, then update all polarization histories. Do not apply
an instantaneous harmonic correction after the existing volume ADE step.
Continuation, memory accounting, source-material guards, periodic seams, and
explicit unsupported-backend errors must include these new interface states.
Ohmic conductivity in a mixed interface needs the same constitutive treatment;
reject unsupported combinations rather than silently leaving it volume-averaged.

### Acceptance tests and rollout

- Compare the discrete-time constitutive transfer response with the analytical
  diagonal mixture evaluated under the trapezoidal frequency mapping, including
  fractions 0/1, equal materials, static limits, and shared epsilon-infinity.
- Cover Lorentz dielectrics, Drude metals, dielectric/dispersive and
  dispersive/dispersive pairs, and interfaces near a normal-response resonance.
  Check passivity and long-time stability; do not assume component stability alone
  proves stability of the complete spatially discretized solver.
- Test slab reflection/transmission at normal and oblique incidence, both TE and
  TM. Normal incidence on a flat interface has tangential E and cannot validate
  the harmonic branch by itself. Use rotated interfaces or a validated oblique
  excitation fixture; the present rectilinear PlaneWaveSource is normal-incidence
  only, and nonzero-phase Bloch boundaries remain unsupported.
- Sweep interface position through a cell, test curved geometry and grid
  refinement, and verify continuation and periodic seam equivalence. Preserve
  constant-material and explicit-volume results.
- After these checks pass, honor `farjadpour_diagonal` for dispersive scenes by
  default and remove the forced-volume bypasses. Keep `volume` selectable, rerun
  the matched CMOS device/reference pair on the RTX3090, and publish the measured
  difference. Keep detector-plane flux colocation as an independent validation:
  constitutive averaging does not automatically repair monitor interpolation.

The theoretical direction follows [Farjadpour et al. (2006)](https://doi.org/10.1364/OL.31.002972)
and the extension to dispersive media is supported by
[Deinega and Valuev (2007)](https://doi.org/10.1364/OL.32.003429).
The realization above is a proposed implementation to derive and validate, not
an assertion that this code already implements either paper's complete scheme.

## RTX3090 execution budget

Use broadband pulses for every case; no per-wavelength simulations. Reuse existing
results and begin with inexpensive planar controls. After those pass, start with
one 510 fs baseline and one depth-refined device/reference pair; add boundary or
component controls only when the preceding result justifies them. Run large jobs
sequentially, estimate fields plus ADE/DFT memory before launching, and target
under 20 GiB of allocated GPU memory. Save actual grids, timesteps, material
hashes, raw fluxes, stop-time diagnostics, and calibrated spectra for every case.
The numerical thresholds above are proposed accuracy gates, not measured results.
