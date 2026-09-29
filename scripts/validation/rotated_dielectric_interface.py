"""Source-free TE pulse reflection from a rotated, lossless dielectric plane.

The wave travels normal to the physical interface at either grid orientation.
An infinite transverse pulse avoids finite-beam angular averaging. A central
1 µm strip is observed before radiation from the distant boundaries can reach
it. A matching empty-domain pulse calibrates the characteristic-field energy.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import beamz as bz
from beamz.design.raster import (
    ExtrudedPolygon,
    Grid,
    Object,
    Polygon,
    RasterOptions,
    Scene,
)
from beamz.lattice import component_coordinates_rectilinear


def run(dx_nm, angle, smoothing, *, dielectric):
    dx = dx_nm * 1e-9
    theta = np.deg2rad(angle)
    normal = np.array([np.cos(theta), np.sin(theta)])
    tangent = np.array([-normal[1], normal[0]])
    size = 10e-6
    cells = round(size / dx)
    grid = Grid.uniform(
        (-size / 2, -size / 2, 0), (size / 2, size / 2, dx), (cells, cells, 1)
    )
    lower = 0.17 * dx
    vertices = tuple(
        tuple(s * normal + t * tangent)
        for s, t in ((lower, -size), (2 * size, -size), (2 * size, size), (lower, size))
    )
    scene = Scene(
        (bz.Material(1), bz.Material(2.25)),
        (Object(ExtrudedPolygon(Polygon(vertices), -dx, 2 * dx), 1),)
        if dielectric
        else (),
    )
    dt = 0.6 * dx / (bz.LIGHT_SPEED * np.sqrt(2))
    steps = round(10e-15 / dt)
    sim = bz.Simulation(
        scene=scene,
        raster_grid=grid,
        polarization="te",
        raster_options=RasterOptions(smoothing=smoothing),
        time=np.arange(steps + 2) * dt,
        boundaries=[bz.PML(thickness=0.6e-6, formulation="cpml")],
    )
    state = sim.initial_state()
    impedance = np.sqrt(bz.MU_0 / bz.EPS_0)
    fields = {}
    for component, factor in (
        ("Ex", tangent[0]),
        ("Ey", tangent[1]),
        ("Hz", 1 / impedance),
    ):
        coords = component_coordinates_rectilinear(
            component, grid, plane="xy", polarization="te"
        )
        s = normal[0] * coords["x"][None, :] + normal[1] * coords["y"][:, None]
        # H is initialized half a timestep behind E.
        if component == "Hz":
            s = s + bz.LIGHT_SPEED * dt / 2
        profile = np.exp(-0.5 * ((s + 1.2e-6) / 0.4e-6) ** 2) * np.cos(
            2 * np.pi * (s + 1.2e-6) / 550e-9
        )
        fields[component.lower()] = jnp.asarray(factor * profile, dtype=jnp.float32)
    state = state._replace(**fields)
    final = sim.advance(
        state=state, num_steps=steps, backend="jax", performance=False
    ).state
    next_state = sim.advance(
        state=final, num_steps=1, backend="jax", performance=False
    ).state
    ex = 0.5 * (np.asarray(final.ex)[1:, :] + np.asarray(final.ex)[:-1, :])
    ey = 0.5 * (np.asarray(final.ey)[:, 1:] + np.asarray(final.ey)[:, :-1])
    h = 0.5 * (np.asarray(final.hz) + np.asarray(next_state.hz))
    electric = tangent[0] * ex + tangent[1] * ey
    characteristic = 0.5 * (electric + (-1 if dielectric else 1) * impedance * h)
    centers = 0.5 * (grid.x_edges[:-1] + grid.x_edges[1:])
    s = normal[0] * centers[None, :] + normal[1] * centers[:, None]
    t = tangent[0] * centers[None, :] + tangent[1] * centers[:, None]
    mask = (abs(t) < 0.5e-6) & ((s < -0.2e-6) if dielectric else (s > 0.2e-6))
    assert np.isfinite(characteristic).all()
    return float(np.sum(characteristic[mask].astype(float) ** 2) * dx**2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx-nm", type=float, nargs="+", default=[20, 10, 5])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/reviews/cmos-isolation/periodic-interfaces/rotated-fresnel.json"
        ),
    )
    args = parser.parse_args()
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("Local GPU required")
    records = []
    for dx in args.dx_nm:
        for angle in (0, 45):
            reference = run(dx, angle, "volume", dielectric=False)
            for smoothing in ("volume", "farjadpour_diagonal"):
                reflected = run(dx, angle, smoothing, dielectric=True)
                value = reflected / reference
                record = dict(
                    dx_nm=dx,
                    interface_angle_degrees=angle,
                    smoothing=smoothing,
                    R=value,
                    R_exact=0.04,
                    R_error_pp=100 * (value - 0.04),
                )
                print(json.dumps(record), flush=True)
                records.append(record)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(
                        dict(
                            device=str(jax.devices()),
                            n=1.5,
                            wavelength_nm=550,
                            run_time_fs=10,
                            runs=records,
                        ),
                        indent=2,
                    )
                    + "\n"
                )
                jax.clear_caches()


if __name__ == "__main__":
    main()
