"""Diagnostic closed-box ownership at metal faces; physical dimensions stay fixed.

This changes only Yee sites on a metal box face (within coordinate roundoff).
It does not assume that the proprietary Tidy3D server uses this convention.
"""

from dataclasses import replace

import numpy as np

from beamz.const import EPS_0
from beamz.design.raster import Box, Cylinder, Sphere
from beamz.lattice import component_coordinates_rectilinear


def close_metal_faces(mg, scene, frequency, dt):
    warped = np.tan(np.pi * frequency * dt) / (np.pi * dt)
    metal_ids = [
        i
        for i, m in enumerate(scene.materials)
        if hasattr(m, "eps_model") and m.eps_model(frequency).real < 1
    ]
    if len(metal_ids) != len(mg.dispersion):
        raise ValueError("Metal material order must match the frozen grid")
    tol = 64 * np.finfo(float).eps * max(abs(a).max() for a in mg.grid.edges)
    yee = dict(mg.yee_materials)
    supports = [
        {c: np.array(w) for c, w in weights.items()} for _, weights in mg.dispersion
    ]
    audit = {}
    for component in ("Ex", "Ey", "Ez"):
        coordinates = component_coordinates_rectilinear(component, mg.grid)
        x, y, z = coordinates["x"], coordinates["y"], coordinates["z"]
        xyz = (x[None, None, :], y[None, :, None], z[:, None, None])
        target = np.zeros((len(z), len(y), len(x)), dtype=bool)
        for obj in scene.objects:
            g = obj.geometry
            if obj.material_id not in metal_ids or not isinstance(g, Box):
                continue
            inside = np.ones_like(target)
            face = np.zeros_like(target)
            for a, lo, hi in zip(xyz, g.minimum, g.maximum, strict=True):
                inside &= (a >= lo - tol) & (a <= hi + tol)
                face |= (abs(a - lo) <= tol) | (abs(a - hi) <= tol)
            target |= inside & face
        iz, iy, ix = np.nonzero(target)
        points = np.stack((x[ix], y[iy], z[iz]), axis=1)
        owner = np.full(len(ix), scene.background_material, dtype=np.int32)
        for obj in sorted(scene.objects, key=lambda obj: obj.priority):
            g = obj.geometry
            if isinstance(g, Box):
                inside = np.all(
                    (points >= np.asarray(g.minimum) - tol)
                    & (points <= np.asarray(g.maximum) + tol),
                    axis=1,
                )
            elif isinstance(g, Sphere):
                inside = (
                    np.sum((points - np.asarray(g.center)) ** 2, axis=1) <= g.radius**2
                )
            elif isinstance(g, Cylinder):
                inside = (
                    (
                        np.sum((points[:, :2] - np.asarray(g.center)) ** 2, axis=1)
                        <= g.radius**2
                    )
                    & (points[:, 2] >= g.z_min - tol)
                    & (points[:, 2] <= g.z_max + tol)
                )
            else:
                raise ValueError("This diagnostic supports only the native CMOS shapes")
            owner[inside] = obj.material_id
        values = np.empty(len(ix), dtype=complex)
        for i, m in enumerate(scene.materials):
            eps = (
                m.epsilon_inf
                if i in metal_ids
                else (
                    m.eps_model(warped) if hasattr(m, "eps_model") else m.permittivity
                )
            )
            values[owner == i] = eps
        axis = component[-1].lower()
        changed = 0
        for j, i in enumerate(metal_ids):
            old = supports[j][component][target] > 0.5
            changed += np.count_nonzero(old != (owner == i))
            supports[j][component][target] = owner == i
        eps = np.array(yee["eps_" + axis])
        eps[target] = values.real
        sig = np.array(yee["sig_" + axis])
        sig[target] = 2 * np.pi * warped * EPS_0 * np.maximum(values.imag, 0)
        yee["eps_" + axis], yee["sig_" + axis] = eps, sig
        audit[component] = dict(face_sites=len(ix), changed_metal_sites=int(changed))
    return replace(
        mg,
        yee_materials=yee,
        dispersion=tuple(
            (m, w) for (m, _), w in zip(mg.dispersion, supports, strict=True)
        ),
    ), audit
