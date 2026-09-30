"""CMOS diagnostic: smooth epsilon-infinity but point-sample dielectric response.

Meep documents smoothing only the instantaneous part of dispersive materials.
This tests that distinction using BeamZ's existing diagonal epsilon-infinity;
it does not claim to reproduce Meep's full tensor or Tidy3D's private operator.
"""

from dataclasses import replace

import numpy as np

import beamz as bz
from beamz.design.discretization import MaterialGrid
from beamz.design.raster import Grid, RasterOptions
from beamz.design.raster.engine import rasterize


def point_susceptibility_grid(mg, scene, frequency):
    # Unique positive epsilon labels recover the native point owner in one raster.
    labels = replace(
        scene, materials=tuple(bz.Material(i + 1) for i in range(len(scene.materials)))
    )
    point = MaterialGrid.from_raster_result(
        rasterize(
            labels,
            Grid(*mg.grid.edges),
            options=RasterOptions(periodic_axes=("x", "y")),
            _staircase_materials=tuple(range(len(scene.materials))),
        ),
        dimensions=3,
    )
    indices = [i for i, m in enumerate(scene.materials) if hasattr(m, "eps_model")]
    supports = {i: {} for i in indices}
    metals = tuple((m, w) for m, w in mg.dispersion if m.eps_model(frequency).real < 1)
    for axis in "xyz":
        values = point.yee_materials["eps_" + axis]
        owner = np.rint(values).astype(np.int32) - 1
        np.testing.assert_allclose(values, owner + 1, rtol=0, atol=1e-6)
        if np.any(owner < 0) or np.any(owner >= len(scene.materials)):
            raise ValueError("Point-owner label outside the CMOS material table")
        for dimension in (1, 2):  # Periodic y/x terminal nodes must have one owner.
            if owner.shape[dimension] == mg.shape[dimension] + 1:
                np.testing.assert_array_equal(
                    np.take(owner, 0, axis=dimension),
                    np.take(owner, -1, axis=dimension),
                )
        for medium, weights in metals:
            index = next(i for i, m in enumerate(scene.materials) if m == medium)
            np.testing.assert_allclose(weights["E" + axis], owner == index, atol=1e-7)
        for index in indices:
            supports[index]["E" + axis] = owner == index
    audit = {
        "method": "diagonal epsilon-infinity; point dielectric susceptibility",
        "metal_point_ownership_verified": True,
        "periodic_point_ownership_verified": True,
    }
    return replace(
        mg,
        dispersion=tuple((scene.materials[i], supports[i]) for i in indices),
        dispersion_interfaces=(),
    ), audit


def freeze_point_susceptibilities(mg, scene, frequency, dt):
    from cmos_frequency_frozen import freeze_dielectrics

    point, audit = point_susceptibility_grid(mg, scene, frequency)
    frozen, _ = freeze_dielectrics(point, frequency, dt)
    return frozen, audit
