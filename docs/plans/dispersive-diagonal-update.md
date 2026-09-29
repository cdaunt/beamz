# Causal diagonal interface update

The scalar `PoleResidue` solver now honors the default
`RasterOptions(smoothing="farjadpour_diagonal")`. Explicit `volume` remains
available. This is a diagonal constitutive approximation; it does not implement
full off-diagonal dispersive coupling or Tidy3D's metal staircasing policy.

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

Periodic duplicated faces carry the two physical half-support contributions,
weighted by their actual widths. Their history increments and step coefficients
are added, rather than overwriting one another. Interface states participate in
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
