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


def decompose_normal_air_wave(e0, e1, frequencies, *, dt, spacing, separation):
    """Separate -z/+z waves from two native E planes in uniform-grid air.

    Uses the Yee dispersion relation, not the continuum wave number. Both planes
    must lie in the same uniform air region, away from sources and absorbers.
    Inputs are transverse electric phasors at native E nodes; no interpolation.
    This specialized calibration assumes a normally incident plane wave.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    if (
        not np.isfinite([dt, spacing, separation]).all()
        or min(dt, spacing, separation) <= 0
        or np.any(~np.isfinite(frequencies))
        or np.any(frequencies <= 0)
    ):
        raise ValueError("Calibration needs positive finite frequencies and geometry.")
    argument = spacing * np.sin(np.pi * frequencies * dt) / (bz.LIGHT_SPEED * dt)
    if np.any(frequencies * dt >= 0.5) or np.any(argument >= 1):
        raise ValueError("Calibration frequencies must propagate below the Yee cutoff.")
    k = 2 * np.arcsin(argument) / spacing
    phase = np.exp(1j * k * separation)
    denominator = 1 / phase - phase
    if np.any(abs(denominator) < 1e-6):
        raise ValueError("Calibration planes give a degenerate wave decomposition.")
    forward = (np.asarray(e1) - phase * np.asarray(e0)) / denominator
    backward = np.asarray(e0) - forward
    return forward, backward, k


def reference_grid(lateral_spacing, depth_refinement=1, max_optical_dz=None):
    saved = np.load(DATA / "tidy3d_reconstructed_grid.npz")
    if not np.isfinite(lateral_spacing) or lateral_spacing <= 0:
        raise ValueError("Lateral spacing must be finite and positive")
    count = int(np.ceil(3e-6 / lateral_spacing - 1e-10))
    xy = np.linspace(-1.5e-6, 1.5e-6, count + 1)
    if depth_refinement not in (1, 2, 4, 8, 16):
        raise ValueError("Depth refinement must be 1, 2, 4, 8 or 16")
    z = saved["z"]
    if max_optical_dz is not None:
        if not np.isfinite(max_optical_dz) or max_optical_dz <= 0:
            raise ValueError(
                "Maximum optical depth spacing must be finite and positive"
            )
        snapshot = json.loads((DATA / "tidy3d_reference_simulation.json").read_text())
        geometries = {o["name"]: o["geometry"] for o in snapshot["structures"]}
        silicon = geometries["silicon layer"]
        lower = (silicon["center"][2] + silicon["size"][2] / 2) * 1e-6
        upper = (
            max(
                g["center"][2] + g["radius"]
                for name, g in geometries.items()
                if name.startswith("len ")
            )
            * 1e-6
        )
        refined = []
        for a, b in zip(z[:-1], z[1:], strict=True):
            inside = b > lower + 1e-18 and a < upper - 1e-18
            count = (
                max(1, int(np.ceil((b - a) / max_optical_dz - 1e-10))) if inside else 1
            )
            refined.append(np.linspace(a, b, count, endpoint=False))
        z = np.concatenate(refined + [z[-1:]])
    if depth_refinement > 1:
        z = np.concatenate(
            [
                np.linspace(a, b, depth_refinement, endpoint=False)
                for a, b in zip(z[:-1], z[1:], strict=True)
            ]
            + [z[-1:]]
        )
    return Grid(xy, xy, z)


def reference_scene(grid, stage="full"):
    snapshot = json.loads((DATA / "tidy3d_reference_simulation.json").read_text())
    materials = [bz.Material(1)]
    ids = {}
    objects = []
    lower = np.array([a[0] for a in grid.edges])
    upper = np.array([a[-1] for a in grid.edges])
    if stage not in {
        "full",
        "no_lenses",
        "bare_stack",
        "planar_red",
        "planar_green",
        "planar_blue",
    }:
        raise ValueError(f"Unknown component stage: {stage}")
    for item in snapshot["structures"]:
        name = item["name"]
        if stage != "full" and name.startswith("len "):
            continue
        if (stage == "bare_stack" or stage.startswith("planar_")) and (
            name.startswith("metal") or name.startswith("interconnect")
        ):
            continue
        planar_filter = stage.startswith("planar_") and name.endswith(" filter")
        if (
            planar_filter
            and name
            != {
                "planar_red": "red filter",
                "planar_green": "green1 filter",
                "planar_blue": "blue filter",
            }[stage]
        ):
            continue
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
            if planar_filter:
                center[:2] = (lower[:2] + upper[:2]) / 2
                size[:2] = upper[:2] - lower[:2]
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
    stage="full",
    depth_refinement=1,
    max_optical_dz=None,
):
    grid = reference_grid(lateral_spacing, depth_refinement, max_optical_dz)
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
        scene=Scene((bz.Material(1),)) if reference else reference_scene(grid, stage),
        raster_grid=grid,
        raster_options=RasterOptions(smoothing=smoothing),
        sources=sources,
        monitors=monitors,
        boundaries=boundaries,
        time=np.arange(0.0, run_time + 0.5 * dt, dt),
    )
