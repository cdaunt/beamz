"""Offline reference geometry and reconstructed Tidy3D depth-grid control.

The recorded mesh was reconstructed with the snapshot's Tidy3D version. No
Tidy3D installation or cloud execution is needed to run this BeamZ control.
"""

import json
from pathlib import Path

import numpy as np

import beamz as bz
from beamz.design.raster import (
    Box,
    Cylinder,
    Grid,
    Object,
    RasterOptions,
    Scene,
    Sphere,
)

DATA = Path(__file__).resolve().parent / "data/cmos_rgb"


def reference_grid(lateral_spacing):
    saved = np.load(DATA / "tidy3d_reconstructed_grid.npz")
    if not np.isfinite(lateral_spacing) or lateral_spacing <= 0:
        raise ValueError("Lateral spacing must be finite and positive")
    count = int(np.ceil(3e-6 / lateral_spacing - 1e-10))
    xy = np.linspace(-1.5e-6, 1.5e-6, count + 1)
    return Grid(xy, xy, saved["z"])


def reference_scene(grid):
    snapshot = json.loads((DATA / "tidy3d_reference_simulation.json").read_text())
    materials = [bz.Material(1)]
    ids = {}
    objects = []
    lower = np.array([a[0] for a in grid.edges])
    upper = np.array([a[-1] for a in grid.edges])
    for item in snapshot["structures"]:
        medium = item["medium"]
        key = json.dumps(medium, sort_keys=True)
        if key not in ids:
            ids[key] = len(materials)
            materials.append(
                bz.PoleResidue(
                    medium["eps_inf"],
                    [(complex(**a), complex(**c)) for a, c in medium["poles"]],
                    frequency_range=medium["frequency_range"]
                    or (bz.LIGHT_SPEED / 700e-9, bz.LIGHT_SPEED / 400e-9),
                )
            )
        raw = item["geometry"]
        center = np.asarray(raw["center"]) * 1e-6
        if raw["type"] == "Box":
            size = np.asarray(raw["size"], dtype=float) * 1e-6
            geometry = Box(
                tuple(np.maximum(lower, center - size / 2)),
                tuple(np.minimum(upper, center + size / 2)),
            )
        elif raw["type"] == "Sphere":
            geometry = Sphere(tuple(center), raw["radius"] * 1e-6)
        elif raw["type"] == "Cylinder" and raw["axis"] == 2:
            half = raw["length"] * 0.5e-6
            geometry = Cylinder(
                tuple(center[:2]),
                raw["radius"] * 1e-6,
                center[2] - half,
                center[2] + half,
            )
        else:
            raise ValueError(f"Unsupported reference geometry: {raw['type']}")
        objects.append(Object(geometry, ids[key], len(objects)))
    return Scene(tuple(materials), tuple(objects))


def make_reference_mesh_sim(
    *,
    lateral_spacing,
    sources,
    monitors,
    run_time,
    courant=0.7,
    reference=False,
    smoothing="farjadpour_diagonal",
    boundary="absorber",
):
    grid = reference_grid(lateral_spacing)
    snapshot = json.loads((DATA / "tidy3d_reference_simulation.json").read_text())
    half_size = snapshot["size"][2] * 0.5e-6
    thicknesses = (-half_size - grid.z_edges[0], grid.z_edges[-1] - half_size)
    mesh_record = json.loads((DATA / "tidy3d_reconstructed_grid.json").read_text())
    reference_dt = mesh_record["dt"]
    dt = min(grid.cfl_time_step(courant), reference_dt)
    boundaries = [bz.Periodic(axes=("x", "y"))]
    for edge, side, thickness in zip(
        ("front", "back"), ("minus", "plus"), thicknesses, strict=True
    ):
        parameters = snapshot["boundary_spec"]["z"][side]["parameters"]
        # Tidy3D declares conductivity in units of 2*epsilon0/reference_dt.
        sigma = parameters["sigma_max"] * 2 * bz.EPS_0 / reference_dt
        boundaries.append(
            bz.PML(edges=(edge,), thickness=thickness, formulation="cpml")
            if boundary == "cpml"
            else bz.Absorber(
                edges=(edge,),
                thickness=thickness,
                sigma_max=sigma,
                m=parameters["sigma_order"],
            )
        )
    return bz.Simulation(
        scene=Scene((bz.Material(1),)) if reference else reference_scene(grid),
        raster_grid=grid,
        raster_options=RasterOptions(smoothing=smoothing),
        sources=sources,
        monitors=monitors,
        boundaries=boundaries,
        time=np.arange(0.0, run_time + 0.5 * dt, dt),
    )
