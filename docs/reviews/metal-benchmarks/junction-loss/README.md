# Junction and loss isolation

This follow-up tests two hypotheses on the small resonant cell before changing the CMOS solver. All FDTD runs use the local RTX3090, the saved gold fit, the existing source and boundaries, and 300 fs runtime.

## Experiments

1. Validate direct material absorption against native Yee flux in the flat multilayer.
2. Repeat on the central antenna (132.307692 nm) on both the reconstructed notebook mesh and the 5 nm mesh.
3. Preserve the binary metal pole ownership and change only nonmetal-owned supports containing metal and at least two dielectrics. Exclude metal from the volume fractions, renormalize the remaining dielectric fractions, and average tangential permittivity arithmetically and normal permittivity harmonically. Here the surviving dielectric interface is horizontal. This is a diagnostic alternative, not a proposed general junction law or an asserted reproduction of Tidy3D.
4. Translate the physical device by a quarter cell relative to the fixed 5 nm lattice. Continuum reflectivity should be invariant. This separates grid registration from the previous refinement and junction experiments.
5. Change production defaults only if these controls demonstrate a defect and a validated remedy. Carry a demonstrated remedy to CMOS; do not tune its materials to fit an image.

## Independent absorption

`metal_junction_loss.py` adds diagnostic native field gathers through a scoped monitor-compiler hook. It does not alter the time stepping, ADE, source, boundaries, or production monitor API. Material absorption uses E at its actual component locations, without interpolating across an interface. The same `weight > 1e-7` active-material threshold as ADE compilation excludes roundoff fractions at geometric boundaries. For this staircased test, the retained weights are all one.

For angular frequency ω and step Δt, define

- Ω = 2 sin(ω Δt / 2) / Δt, the Yee discrete frequency;
- f* = tan(ω Δt / 2) / (π Δt), the trapezoidal ADE frequency;
- Q = Σ(component, metal sites) V |E|², with primal widths along the component and dual widths transversely. Periodic endpoint half-volumes avoid counting duplicated boundary samples twice.

Absorbed power is ε₀ Ω Im[ε(f*)] Q / 2. The saved dimensionless absorption divides this by the empty-run incident power, using exactly the same DFT normalization. This expression follows from the discrete frequency-domain Maxwell/ADE equations; using the continuous ε(f) and frequency without accounting for the discrete update would introduce an additional approximation.

The independent local energy check uses native H at a z-cell midpoint and the mean of the adjacent native tangential E values. H's half-step DFT phase is corrected by exp(−i ω Δt/2). Transverse native quadrature retains all spatial variation, including evanescent fields; averaging the field across the plane first would discard their contribution. The net downward flux above the device minus transmitted flux below it must equal the material loss. This checks the implemented discrete solution; it does not certify continuum accuracy.

Outgoing reflection remains normalized to the matched empty run. `1−R−T−A` and the local balance residual are reported separately: source-region/return-wave normalization effects can affect the former without implying nonconservation inside the device.

## Tidy3D documentation limit

[Tidy3D's subpixel specification](https://docs.flexcompute.com/projects/tidy3d/en/v2.10.2/api/discretization/_autosummary/tidy3d.SubpixelSpec.html) distinguishes dielectric polarized averaging and metal staircasing. [Its implementation documentation](https://docs.flexcompute.com/projects/tidy3d/en/stable/_modules/tidy3d/components/subpixel_spec.html) describes polarized averaging as an approximation of contour-path averaging. Those names do not specify the treatment of every three-material support or boundary tie. The newer [local subpixel extras](https://docs.flexcompute.com/projects/tidy3d/en/v2.10.0/extras/index.html) could expose effective permittivity locally, but require authentication; no credentials or original task arrays were available for this comparison. The checked point-ownership export is therefore not a server-operator equivalence check.

## Reproduction

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1
.venv/bin/python scripts/validation/metal_junction_loss.py --length 0
for mesh in reference refined; do
  for variant in baseline polarized; do
    .venv/bin/python scripts/validation/metal_junction_loss.py --mesh "$mesh" --variant "$variant"
  done
done
.venv/bin/python scripts/validation/metal_grid_registration.py --z-shift 1.25
.venv/bin/python scripts/validation/metal_grid_registration.py --z-shift -1.25
.venv/bin/python scripts/validation/metal_grid_registration.py --xy-shift 1.25
```

Results and checks are summarized in the parent executed notebook and `junction_loss_summary.csv`. These controls address the gold resonator. They do not resolve the separate aluminum-sphere flux/optical-theorem discrepancy or prove a cause of the CMOS red/blue gap.

The lateral translation is repeated at 2.5 nm with a 0.625 nm shift:

```bash
.venv/bin/python scripts/validation/metal_grid_registration.py --dx 2.5 --xy-shift 0.625
JAX_PLATFORMS=cpu .venv/bin/python scripts/validation/analyze_metal_junction_loss.py
```

## Completed results

| Mesh | Junction treatment | R (%) | Direct metal A (%) | Local balance residual (percentage points) |
|---|---|---:|---:|---:|
| 5 nm | baseline | 83.869152 | 16.126130 | 0.00003259 |
| Planar reference | baseline | 98.469652 | 1.574481 | 0.00000530 |
| Reconstructed | baseline | 82.149551 | 17.700278 | 0.00004585 |
| 5 nm | polarized | 83.752078 | 16.243504 | 0.00000022 |
| Reconstructed | polarized | 82.126292 | 17.764218 | 0.00006633 |

The junction variant is too small to close the reference gap. Loss is real absorption of the **discrete simulated field**, with no demonstrated multi-point extraction error. The quarter-cell lateral-registration sensitivity decreases from 1.536 points at 5 nm to 0.984 points at 2.5 nm, which identifies a remaining spatial-discretization concern. Neither control proves the cause of the complete Tidy3D/CMOS discrepancy. Defaults remain dielectric diagonal averaging and metal staircasing.

All nine notebook code cells execute successfully. The saved-data analyzer asserts passive component absorption, local conservation within 0.001 percentage points, late field-DFT drift below 0.01%, and the tighter global balance on the 5 nm cases. Measured local residuals are substantially below the assertion threshold. Raw native field files are reproducible, ignored local intermediates; final spectra, reports, and summaries are committed.
