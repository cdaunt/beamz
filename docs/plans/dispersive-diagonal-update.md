# Causal diagonal interface update

The scalar `PoleResidue` solver defaults to diagonal Farjadpour averaging for
**dielectrics**, and **staircasing for metal interfaces**. The diagonal law below
is unchanged; it is not the full off-diagonal dispersive tensor from the paper.

## Material-dependent defaults

```python
RasterOptions()  # dielectric: farjadpour_diagonal; metal: staircase
RasterOptions(metal_smoothing="inherit")  # previous all-material smoothing
RasterOptions(reference_frequency=5.5e14)  # explicit classification frequency, Hz
```

A scalar medium is metallic when Re(εr) < 1 at the reference frequency. Static
tensors use their smallest principal permittivity. Conductivity alone does not
make a positive-permittivity dielectric an optical metal. Simulation selects the
mean of its positive source center frequencies unless explicitly overridden;
without such a source, standalone rasterization uses each dispersive material's
fit-band midpoint. Classification is made once, not separately per DFT frequency.
Materials that cross the threshold within a broad band should use an explicit
reference frequency for reproducible comparisons.

At a support touching metal, the native rasterizer selects the material at the
actual Yee location using geometric painter order and half-open boundaries.
At the domain's final faces it uses the interior limit, so a metal box clipped
to the domain does not acquire artificial air samples at a periodic seam.
On nonuniform grids this location can differ from the support-volume midpoint.
The same decision applies to ε∞, conductivity and **every** constituent ADE
indicator; no fractional host polarization or harmonic interface branch remains
at a staircased support. A direct point check also catches small metal features
missed by adaptive volume integration. Interfaces involving only dielectrics
retain diagonal averaging, including its existing ambiguous-normal fallback.

Classification retains original material IDs even when ε∞ is identical.
Persistent raster cache identity includes the metal IDs, policy and reference
frequency; Design's in-memory cache also tracks source-dependent classification.
`metal_smoothing="inherit"` uses the selected `smoothing` for all media, including
explicit `volume`, and reproduces the historical all-diagonal controls.

The policy follows Tidy3D's documented dielectric/metal defaults, but its
`PolarizedAveraging` is a contour-path approximation: identical policy names do
not establish identical dielectric operators. Full dispersive tensor smoothing
remains unsupported. See the [paper audit](../reviews/cmos-reverse-engineering/paper-interface-audit.md).

## Constitutive law

For geometric fractions f_j and interface-normal component n_i, define

```text
A(s) = sum_j f_j epsilon_j(s)
H(s) = 1 / sum_j f_j / epsilon_j(s)
epsilon_ii(s) = (1 - n_i^2) A(s) + n_i^2 H(s).
```

The native rasterizer supplies laminar interface normals and fractions on each
actual electric Yee support, including nonuniform physical support volumes.
Material identity is preserved even when different dispersive media have equal
static epsilon-infinity. Ambiguous interfaces retain the native volume fallback.

Tangential polarization uses the existing bulk ADE with occupancy multiplied by
`1 - n_i^2`. The normal branch stores constituent polarization histories sparsely
at mixed supports. Its constituents share normal D, while their fraction-weighted
E fields sum to the macroscopic E. There is no harmonic averaging of residues,
frequency-by-frequency fitting, or extra wavelength simulation.

## Relation to inverse-tensor smoothing

This implementation diagonalizes the effective **permittivity** tensor before
inversion. It is not the diagonal of the inverse tensor in Deinega and Valuev
(2007), Eq. (1): in general, `1 / [(1-w) A + w H]` differs from
`(1-w)/A + w/H`. They coincide at the pure normal/tangential limits. Neither
diagonal truncation retains the full cross-component coupling required by that
paper's method. This distinction is especially important at curved metal
interfaces; see the [paper audit](../reviews/cmos-reverse-engineering/paper-interface-audit.md)
for numerical examples and proposed physical validation gates.

## Trapezoidal elimination

For each constituent pole p and residue c, in relative-polarization units,

```text
a = (2 + dt*p) / (2 - dt*p)
b = dt*c / (2 - dt*p)
q_new = a*q_old + b*(E_local_old + E_local_new)
r_j = 2 Re sum_k b_jk
H_inf = 1 / sum_j f_j / epsilon_inf,j
H_step = 1 / sum_j f_j / (epsilon_inf,j + r_j)
P_j = 2 Re sum_k q_jk
P_normal = H_inf sum_j f_j P_j / epsilon_inf,j
D_normal_old / epsilon0 = H_inf E_old + P_normal
E_local_old,j = (D_normal_old / epsilon0 - P_j) / epsilon_inf,j
h_j = 2 Re sum_k a_jk q_jk + r_j E_local_old,j
h_normal = H_step sum_j f_j h_j / (epsilon_inf,j + r_j)
delta = n_i^2 (H_step - H_inf)
```

Add `delta` to the common implicit electric-step denominator. The corresponding
history increment is `n_i^2 (h_normal - P_normal) - delta E_old`; together with
the denominator's old-field term this produces the normal polarization change.
After solving the common E step, recover

```text
D_normal_new / epsilon0 = H_step E_new + h_normal
E_local_new,j = (D_normal_new / epsilon0 - h_j) / (epsilon_inf,j + r_j)
```

and advance each q. This realizes the harmonic rational response without finding
its effective poles. Tests compare the response at the trapezoidal mapped
frequency `tan(omega*dt/2)/(pi*dt)`; continuous-frequency error is a separate
time-discretization question.

Periodic supports now join **material fractions before smoothing**. Native
rasterization integrates the wrapped pieces of the full dual volume, using the
actual first/last cell widths, then constructs one effective constitutive law.
Averaging two already-smoothed halves is incorrect for a harmonic response.
Duplicate faces use the same wrapped support and metal point owner. Curvature
classification uses the complete support rather than its artificially narrower
pieces. `Simulation` infers raster periodic axes from its boundaries; standalone
rasters can set `RasterOptions(periodic_axes=("x", "y"))`. Cache identity includes
these axes. Explicit precomputed material grids remain responsible for their
own material geometry.

The compiler retains width-weighted joins for legacy static voxel coefficients
and externally supplied dispersive supports. For a native periodic raster the
duplicate supports are already equal. Interface history increments and step
coefficients are added rather than overwriting one another. Interface states participate in
continuation checks, memory estimates, cache identity, and source-sheet guards.
Uniform-grid volume dispersion also consumes the native Yee epsilon-infinity
arrays: reconstructing them from cell averages would give the instantaneous and
pole responses different material occupancies.

## Supported scope

- Scalar passive pole-residue constituents, including Drude and Lorentz media;
  static scalar dielectric neighbors are allowed.
- JAX single-device uniform and rectilinear grids, 2D and 3D.
- Intrinsic tensor constituents and separate Ohmic conductivity at polarized
  dispersive interfaces are rejected. Represent physical loss with poles.
- Full-tensor dispersion, native CUDA/multi-device dispersion and dispersive
  energy-based termination remain unsupported.
- This update does not change flux-monitor interpolation across a discontinuity.
  Spatial convergence, boundary reflection and interface-plane measurement error
  must still be checked independently.

See [the implementation and comparison plan](cmos-reference-agreement.md) and
[local measurements](../reviews/cmos-polarized/README.md).
