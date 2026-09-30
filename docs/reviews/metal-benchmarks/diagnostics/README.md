# Auxiliary and superseded controls

These directories are not the primary convergence data.

- `all-side-cpml/` and `default-alpha-periodic/` used the automatic CPML alpha before the empty-domain illumination check identified large reflection. Their sphere comparisons are excluded from the final Mie convergence series. The final all-side control uses explicit zero alpha and lives in the parent directory.
- `prior_reflection_measurements/` retains below-source empty-field subtraction and intermediate native-plane extraction. On the reconstructed graded mesh, subtracting the empty-grid reflected amplitude introduced a coherent bias. The final above-source outgoing measurement passes the planar analytic control. These archived runs also retain the original SDK-versus-snapshot material-fit comparison on finer meshes.
- `dielectric_volume_cpu/` is a valid supplementary fixed-mesh control. Arithmetic dielectric volume averaging gives the same central-cell reflection as the diagonal GPU run; the compiled epsilon and metal pole-owner arrays are exactly equal. This checks the two BeamZ policies, not Tidy3D's server-side interface operator.

See the [main report](../README.md) for limitations and reproduction commands. Raw surface field arrays are local, ignored intermediates; the committed spectra and JSON records are retained for audit.
