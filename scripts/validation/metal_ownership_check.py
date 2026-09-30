"""Compare BeamZ metal ownership to exported Tidy3D client point samples.

First run export_metal_reference_grid.py with tidy3d==2.9.0. This checks point
sampling, not the proprietary server's subpixel constitutive operator.
"""

import json

import numpy as np
from metal_benchmarks import DATA, OUT, material

import beamz as bz
from beamz.design.raster import Box, Grid, Object, RasterOptions, Scene


def main():
    saved = np.load(DATA / "reflector_grid_132.308nm.npz")
    x, y, z = (saved[a] for a in "xyz")
    length = 132.3076923076923e-9
    metal = material("reflector_gold_snapshot")
    scene = Scene(
        (bz.Material(1), metal, bz.Material(1.892), bz.Material(2.25)),
        (
            Object(Box((-60e-9, -150e-9, z[0]), (60e-9, 150e-9, -130e-9)), 3),
            Object(Box((-60e-9, -150e-9, -130e-9), (60e-9, 150e-9, 0)), 1),
            Object(Box((-60e-9, -150e-9, 0), (60e-9, 150e-9, 50e-9)), 2),
            Object(Box((-45e-9, -length / 2, 50e-9), (45e-9, length / 2, 80e-9)), 1),
        ),
    )
    simulation = bz.Simulation(
        scene=scene,
        raster_grid=Grid(x, y, z),
        raster_options=RasterOptions(reference_frequency=bz.LIGHT_SPEED / 850e-9),
        boundaries=[bz.Periodic(axes=("x", "y"))],
        time=np.arange(2) * 1e-18,
    )
    reference = np.load(DATA / "reflector_point_epsilon.npz")
    rows = []
    for component in ("Ex", "Ey", "Ez"):
        actual = np.asarray(
            simulation.material_grid.yee_materials["eps_" + component[-1].lower()]
        )
        expected = reference[component]
        # Both grids use E along its cell-center coordinate, transverse edges.
        # BeamZ retains the final transverse boundary faces; Tidy drops them.
        actual = actual[: expected.shape[0], : expected.shape[1], : expected.shape[2]]
        beamz_mask = np.isclose(actual, metal.epsilon_inf)
        tidy_mask = np.isclose(expected, metal.epsilon_inf)
        mismatch = beamz_mask != tidy_mask
        rows.append(
            dict(
                component=component,
                beamz_count=int(beamz_mask.sum()),
                tidy3d_point_count=int(tidy_mask.sum()),
                mismatch_count=int(mismatch.sum()),
            )
        )
    (OUT / "metal_point_ownership.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    assert all(row["mismatch_count"] == 0 for row in rows)
    volume = bz.Simulation(
        scene=scene,
        raster_grid=Grid(x, y, z),
        raster_options=RasterOptions(
            smoothing="volume", reference_frequency=bz.LIGHT_SPEED / 850e-9
        ),
        boundaries=[bz.Periodic(axes=("x", "y"))],
        time=np.arange(2) * 1e-18,
    )
    diagonal_grid, volume_grid = simulation.material_grid, volume.material_grid
    differences = {
        component: float(
            np.max(
                abs(
                    np.asarray(diagonal_grid.yee_materials[component])
                    - np.asarray(volume_grid.yee_materials[component])
                )
            )
        )
        for component in ("eps_x", "eps_y", "eps_z")
    }
    for i, ((_, left), (_, right)) in enumerate(
        zip(diagonal_grid.dispersion, volume_grid.dispersion, strict=True)
    ):
        for component in ("Ex", "Ey", "Ez"):
            differences[f"pole_owner_{i}_{component}"] = float(
                np.max(abs(np.asarray(left[component]) - np.asarray(right[component])))
            )
    (OUT / "dielectric_volume_coefficients.json").write_text(
        json.dumps(differences, indent=2) + "\n"
    )
    print("Diagonal versus volume coefficient maximum differences:", differences)


if __name__ == "__main__":
    main()
