# Metal controls on the local RTX3090

The smaller examples reproduce the **remaining inter-solver discrepancy** without the complete CMOS sensor. They also reveal two benchmark pitfalls: a material-library fit change and a CPML default that fails this strongly graded mesh. Neither is an established explanation of the original CMOS discrepancy.

[Executed notebook](https://github.com/beamzorg/beamz/blob/t3code/add-cmos-rgb-notebook/examples/notebooks/metal_solver_benchmarks.ipynb) · [GPU runner](https://github.com/beamzorg/beamz/blob/t3code/add-cmos-rgb-notebook/scripts/validation/metal_benchmarks.py) · [Sphere source](https://www.flexcompute.com/tidy3d/examples/notebooks/PlasmonicNanoparticle/) · [Metasurface source](https://www.flexcompute.com/tidy3d/examples/notebooks/GradientMetasurfaceReflector/)

## What the controls establish

1. **Bulk planar response passes a useful check.** For the air/MgF₂/gold/silica stack, analytical reflectivity is **98.460108%**. BeamZ gives **98.469652%** on the reconstructed Tidy3D mesh, **98.443857%** on the 5 nm mesh, and **98.451454%** on the 2.5 nm mesh. This is a power check at 850 nm, not certification of all material/interface behavior.
2. **The resonant metal–dielectric cell still differs.** At antenna length 132.307692 nm, Tidy3D's plot reads approximately **87.415%** reflectivity. BeamZ gives **82.150%** on the reconstructed notebook mesh, **83.869%** at 5 nm, and **83.265%** at 2.5 nm. The fine-grid gap is **−4.150 percentage points**, well beyond the approximate ±0.2-point image-reading allowance. Refining this grid does not close it. At this length, the 5 nm relative phase differs by only **0.503°**, while the reconstructed-mesh phase differs by **25.529°**; at 280 nm, the 5 nm phase still differs by **10.646°**. A single correction to power normalization would not establish full phase/power parity.
3. **Material names are insufficient provenance.** The saved notebook gold fit has ε(850 nm) = **−29.451832 + 1.025111i**; the current SDK fit has **−28.624676 + 1.251599i**. The saved fit has 18.1% less imaginary permittivity. Switching to it changes central reflectivity by roughly 4.3 points in the initial fine-grid controls, but leaves a substantial gap. The actual CMOS material snapshot was already imported; this discovery does not demonstrate a corresponding CMOS input error.
4. **Curved staircased metal requires refinement.** The two sphere controls provide an independent analytical target. Their completed numerical errors appear below; these runs do not claim sub-percent convergence.

## Sphere convergence

Diameter 10 nm, air background, gold RakicLorentzDrude1998 or the CMOS saved aluminum fit. Our 1/0.5/0.25 nm series has 10/20/40 cells across the diameter; the published Tidy3D sphere example uses 60 (about 0.167 nm). This is a convergence study, not an identical sphere-mesh replication. Errors are max absolute spectral discrepancy divided by the peak exact cross section. Scattering uses the equivalent-current radiation integral on the 8 nm half-width enclosing surface.

| Material | Fine spacing (nm) | Flux absorption error (%) | Optical-theorem absorption error (%) | Projected scattering error (%) |
|---|---:|---:|---:|---:|
| aluminum | 1 | 15.119 | 27.431 | 8.922 |
| aluminum | 0.5 | 9.649 | 15.552 | 6.689 |
| aluminum | 0.25 | 3.784 | 6.495 | 2.647 |
| gold | 1 | 29.032 | 29.949 | 17.173 |
| gold | 0.5 | 17.802 | 18.245 | 10.447 |
| gold | 0.25 | 7.195 | 7.409 | 4.063 |

The primary series uses lateral periodic boundaries with requested period 350 nm and a graded exterior. This approximates an isolated sphere; actual periods are recorded because whole exterior cells are retained. Separate 250 nm-period and all-side CPML controls quantify domain sensitivity in the notebook. Two enclosing surfaces (half-widths 6 and 8 nm), incident impedance, finite fields, and accumulated DFT drift are recorded. Gold 1 nm used a 45 fs checkpoint while later runs use 67.5 fs; its 45→90 fs drift includes the pulse tail and is not a late-time convergence certificate. The other principal sphere runs compare 67.5→90 fs. Direct periodic-array scattering flux is a different observable from isolated-particle Mie scattering; its larger discrepancy is not a demonstrated flux bug. In the electric-dipole approximation, array power per cell divided by isolated radiation with the same dipole moment is `3 λ² / (4 π A)`. At 530 nm the predicted 0.5 nm-grid ratio is 0.514, and the aluminum simulation gives 0.512. The derivation uses the array reflection amplitude `r = i k α / (2 A)` ([Arslan et al., electric-dipole array formulas](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adma.202105868)). The equivalent-current projection extracts free-space radiation of one particle's response, which can still be perturbed by array interactions. Direct-array-power comparisons to the dipole prediction are retained in the reports. Residuals remain: the 0.25 nm gold run differs by up to 25.3% of the predicted array-scattering peak. The array argument therefore explains much of the observable difference but does not certify the direct-flux extraction. Free-space projection and direct flux remain separate diagnostics.

An optical-theorem check obtains extinction from forward scattering and subtracts the projected scattered power. At 0.5 nm it differs from surface-flux absorption by 0.446% of the Mie absorption peak for gold and 5.903% for aluminum. This is an additional extraction/discretization uncertainty, not a passed conservation check; the finest-grid values are retained in the reports. The complex-dipole analytic control validates both radiation power and extinction extraction with errors below 0.04% at the finer surface quadrature.

The local spherical-Bessel Mie implementation agrees with independent `miepython==3.3.0` to approximately 1e−11 or better. The downloadable historical Tidy3D Mie tables differ by 6–7% from matched-model Mie theory. Their unexplained input/normalization discrepancy prevents using those tables as an exact solver reference. Both are retained, with the comparison in `mie_independent_check.json` under the input-data directory.

## Measured sphere-domain sensitivity

- periodic, 0.5 nm mesh, actual lateral span 265.117 nm: absorption change **0.0367%**, projected-scattering change **0.0338%** of the respective Mie spectral peaks versus the 350 nm-requested-period baseline.
- cpml, 1 nm mesh, actual lateral span 902.392 nm: absorption change **0.0167%**, projected-scattering change **0.0173%** of the respective Mie spectral peaks versus the 350 nm-requested-period baseline.

These tests do not compare direct array radiation to isolated radiation, which are different observables.

## Metasurface assumptions and measurement

The checked-in Tidy3D 2.9.0 simulation snapshot contains the original pole fit and subpixel configuration. The cell is 120 × 300 nm; gold antenna width 90 nm and thickness 30 nm; MgF₂ thickness 50 nm and ε=1.892; gold backplane thickness 130 nm; substrate ε=2.25. Excitation is y-polarized at 850 nm. The beamz run uses metal staircasing and diagonal dielectric polarized averaging, with 300 fs runtime. There are three selected antenna lengths, not a dense resonance sweep. The 2.5 nm outgoing-field repeat covers only the planar and central-resonance cases.

`export_metal_reference_grid.py` reconstructs the mesh locally using **Tidy3D 2.9.0**. It makes no cloud call. Metal ownership at the actual E-component coordinates matches Tidy3D's client-side point samples for all three components (zero mismatches at the central antenna length). This does **not** certify the server's subpixel constitutive operator. The reconstructed mesh has a 7.5 nm x step, graded y and z coordinates, and 12 PML cells per outer z boundary; calling it a uniform 7.5 nm grid would be inaccurate.

Final reflectivity uses the outgoing field above the planar TF/SF source. For adjacent native E planes, power is proportional to `−Im(E_low * conj(E_high)) / dz`. The common electromagnetic factors cancel against the empty-domain incident-power reference. Reflected power is measured without subtracting the empty-grid reflected field. Below-source subtraction can coherently bias reflection: an independent discrete 1D Helmholtz solve reproduces that bias on the graded mesh. Final outgoing measurements pass the planar analytic check and agree between neighboring plane pairs. Phase uses the outgoing Ey amplitude relative to the empty-domain incident Ey; curves are anchored separately to the 40 nm antenna. Transmission is a secondary small flux diagnostic, not the certified native-plane reflection measurement.

The BeamZ pulse offset is 4 versus Tidy3D's default 5. Frequency-response normalization and DFT settling checks handle differing time origins; the time traces themselves are not asserted to match. No original Tidy3D task arrays are available. Published curve points are digitized from the saved image with approximate allowances ±0.002 R and ±2°; these are not confidence intervals.

## CPML default finding

On the strongly graded sphere mesh, the default CPML alpha scales inversely with the global timestep. Decreasing interior spacing from 1 to 0.5 to 0.25 nm increases reflected/incident **field amplitude** in the empty-domain control from **15.5% → 49.7% → 81.7%**. Explicit `alpha_max=0` gives about **0.165%** at every spacing, with incident impedance error about 0.33%. The successful benchmarks explicitly select zero alpha, also matching the saved Tidy3D PML default. This does not change BeamZ's global default.

This CPML issue is real for this configuration, but the original CMOS case used a different absorber; previous full-sensor absorber/CPML controls were already small. It is therefore not an explanation of the CMOS mismatch. A future default-policy fix should scale and validate the frequency shift for graded exterior cells rather than merely tuning this case.

## Junction, loss, and registration follow-up

[Detailed method and reproduction](junction-loss/README.md) · [Junction/loss table](junction_loss_summary.csv) · [Registration table](registration_summary.csv)

The junction-only alternative preserves every metal pole-owner coefficient and changes 29 nonmetal electric supports on the reconstructed mesh, or 38 at 5 nm. It shifts central reflectivity by **−0.0233** and **−0.1171 percentage points**, respectively. This does not close the reference gap. All affected supports here are tangential to the dielectric interface; this is not a general validation of a three-material smoothing law.

Direct native-Yee material absorption and native flux through the device agree within **0.000067 percentage points** across the planar control and four resonator runs. The independent integral uses the trapezoidal ADE frequency response and each electric component's support volume. At 5 nm, the baseline gives **R=83.869152%, A=16.126130%, T=0.004369%**. The outgoing-reflection energy residual is only **0.000350 percentage points**. This rules out a several-point loss-extraction artifact for this resonator. The coarse grid has a larger global normalization residual (up to 0.1425 points), while its local energy balance still closes. Conservation of the discrete solution does not establish continuum accuracy. The separate aluminum-sphere flux/optical-theorem uncertainty remains open.

Rigid translations expose remaining discretization sensitivity. Moving the antenna by 1.25 nm in both lateral axes changes R from **83.869152% to 82.333182%** on the 5 nm grid (−1.53597 points). At 2.5 nm, a 0.625 nm lateral shift changes **83.265279% to 82.281302%** (−0.98398 points). Vertical translations of ±1.25 nm at 5 nm change R by −0.21535/+0.08859 points. The continuum reflectivity is invariant under these translations. Thus the staircased resonator is not yet insensitive to grid placement, although the tested shift sensitivity shrinks with refinement. These samples do not bound the full discretization error or explain the full Tidy3D discrepancy.

## What remains to isolate

The tested junction policy is too small an effect, and direct absorption agrees with the discrete flux balance. The remaining concern is the resonant field produced by the effective metal geometry/interface operator: evaluate convergence across grid registrations and compare the complete complex resonance curve against an independent reference. A single aligned-grid resolution sweep can hide placement sensitivity. Tidy3D's client point masks do not establish its server-side effective constitutive coefficients or boundary-tie rules, and a planar power pass does not certify edge fields.

These findings do not demonstrate an algebraic defect in diagonal polarized averaging, establish a production fix, or identify a shared cause for the resonator and original CMOS red/blue differences. Production defaults are unchanged. No full CMOS rerun was launched from this follow-up because it produced no validated solver change to propagate.

## Reproduction

From the repository root, using the GPU-enabled environment:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1
for medium in sphere_gold aluminum; do
  for dx in 1 .5 .25; do
    .venv/bin/python scripts/validation/metal_benchmarks.py sphere --material "$medium" --dx "$dx"
  done
done
.venv/bin/python scripts/validation/metal_benchmarks.py sphere --dx 1 --lateral cpml
.venv/bin/python scripts/validation/metal_benchmarks.py sphere --dx .5 --period 250
for length in 0 40 132.3076923076923 280; do
  .venv/bin/python scripts/validation/metal_benchmarks.py reflector --mesh reference --dx 7.5 --runtime 300 --lengths "$length"
done
.venv/bin/python scripts/validation/metal_benchmarks.py reflector --dx 5 --runtime 300 --lengths 0 40 132.3076923076923 280
.venv/bin/python scripts/validation/metal_benchmarks.py reflector --dx 2.5 --runtime 300 --lengths 0 132.3076923076923
for result in docs/reviews/metal-benchmarks/sphere-*; do
  JAX_PLATFORMS=cpu .venv/bin/python scripts/validation/analyze_metal_sphere.py "$result"
done
JAX_PLATFORMS=cpu .venv/bin/python scripts/validation/analyze_metal_reflector.py
JAX_PLATFORMS=cpu .venv/bin/python scripts/validation/metal_pml_control.py
JAX_PLATFORMS=cpu .venv/bin/python scripts/validation/metal_discrete_planar.py
JAX_PLATFORMS=cpu .venv/bin/python scripts/validation/metal_ownership_check.py
JAX_PLATFORMS=cpu .venv/bin/python scripts/validation/metal_dielectric_variant.py
JAX_PLATFORMS=cpu .venv/bin/python scripts/validation/metal_reference_checks.py
```

The reference checker additionally needs `miepython==3.3.0`; the notebook needs pandas, matplotlib, nbformat, nbclient, and ipykernel. Grid reconstruction additionally needs a separate `tidy3d==2.9.0` environment. Committed inputs suffice for the GPU runs without Tidy3D installed. Run `export_metal_reference_grid.py` in that separate environment to regenerate the reference mesh and point samples. The attached source snapshots and hashes are documented in `examples/data/metal_benchmarks/sources.json`.

Use `execute_metal_notebook.py` to regenerate the executed notebook with the current Python interpreter. The notebook reads saved results; it does not launch simulations. Raw `surface_fields.npz` intermediates are locally retained and ignored by Git because of size; regenerate them with the GPU runner before rerunning sphere field analysis. Final spectra, grids, reports, source snapshots, and plots are committed.

## Audit trail

- Primary data: top-level `sphere-*` and `reflector-*-outgoing` directories.
- `diagnostics/`: excluded initial illumination runs and prior reflection-extraction methods. These do not belong in the final convergence series. Prior SDK-fit runs also document the material-model sensitivity; their fine-grid measurement bias is small but not zero.
- `reflector_checks.json`: planar exact solution, adjacent-plane power and phase checks.
- `discrete_planar_checks.json`: independent discrete outgoing-lead calculation; not an identical CPML/source implementation.
- `metal_point_ownership.json`: client-side metal point-mask comparison.
- `dielectric_volume_coefficients.json`: exact equality of diagonal versus volume ε and pole-owner arrays for the central reconstructed-mesh cell; supplementary CPU simulation under `diagnostics/dielectric_volume_cpu/`.
- `analysis_execution.json`, `provenance.json`: execution environment and validation record.
