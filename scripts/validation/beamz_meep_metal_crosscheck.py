"""BeamZ half of the uniform-grid, common-Drude Meep comparison.

Diagnostic all-interface staircasing matches eps_averaging=False in Meep. The
production dielectric smoothing default is unchanged. Reflection is obtained
from adjacent native E planes above the one-way source.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
from metal_benchmarks import fields, material

import beamz as bz
from beamz.design.discretization import MaterialGrid
from beamz.design.raster import Box, Grid, Object, RasterOptions, Scene
from beamz.design.raster.engine import rasterize
from beamz.lattice import component_coordinates_rectilinear

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/reviews/cmos-parity-followup/beamz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", type=float, default=5)
    parser.add_argument("--length", type=float, default=132.3076923076923)
    parser.add_argument("--coefficients-only", action="store_true")
    parser.add_argument("--meep-ownership", action="store_true")
    args = parser.parse_args()
    out = OUT / f"dx{args.dx:g}-length{args.length:g}"
    if args.meep_ownership:
        out = out.with_name(out.name + "-meep-ownership")
    out.mkdir(parents=True, exist_ok=True)
    dx = args.dx * 1e-9
    grid = Grid(
        *[
            np.linspace(-s / 2, s / 2, round(s / dx) + 1)
            for s in (120e-9, 300e-9, 1600e-9)
        ]
    )
    f0 = bz.LIGHT_SPEED / 850e-9
    original = material("reflector_gold_snapshot")
    epsilon = complex(original.eps_model(f0))
    omega = 2 * np.pi * f0
    gamma = omega * epsilon.imag / (original.epsilon_inf - epsilon.real)
    plasma = np.sqrt((original.epsilon_inf - epsilon.real) * (omega**2 + gamma**2))
    gold = bz.PoleResidue.drude(
        original.epsilon_inf,
        plasma_frequency=plasma,
        damping=gamma,
        frequency_range=(0.9 * f0, 1.1 * f0),
    )
    np.testing.assert_allclose(gold.eps_model(f0), epsilon, rtol=1e-12)
    materials = (bz.Material(1), gold, bz.Material(1.892), bz.Material(2.25))
    objects = [
        Object(Box((-60e-9, -150e-9, -800e-9), (60e-9, 150e-9, -130e-9)), 3),
        Object(Box((-60e-9, -150e-9, -130e-9), (60e-9, 150e-9, 0)), 1),
        Object(Box((-60e-9, -150e-9, 0), (60e-9, 150e-9, 50e-9)), 2),
    ]
    if args.length:
        objects.append(
            Object(
                Box(
                    (-45e-9, -args.length * 0.5e-9, 50e-9),
                    (45e-9, args.length * 0.5e-9, 80e-9),
                ),
                1,
            )
        )
    freq = np.linspace(0.9 * f0, 1.1 * f0, 41)
    z = grid.z_edges
    iz = int(np.argmin(abs(z - 510e-9)))
    ii = int(np.argmin(abs(z - 180e-9)))
    planes = dict(r0=z[iz], r1=z[iz + 1], i0=z[ii], i1=z[ii + 1], t=-300e-9)
    monitors = [
        bz.FluxMonitor(
            center=(0, 0, h), size=(120e-9, 300e-9, 0), freqs=freq, name=n, interval=10
        )
        for n, h in planes.items()
    ]
    source = bz.PlaneWaveSource(
        center=(0, 0, 255e-9),
        size=(120e-9, 300e-9, 0),
        direction="-z",
        pol_angle=np.pi / 2,
        source_time=bz.GaussianPulse(f0, f0 / 3),
    )
    dt = 0.5 * dx / bz.LIGHT_SPEED
    start = time.monotonic()
    results = []
    scenes = (
        (Scene(materials, tuple(objects)),)
        if args.coefficients_only
        else (Scene((bz.Material(1),)), Scene(materials, tuple(objects)))
    )
    for scene in scenes:
        sim = bz.Simulation(
            scene=scene,
            raster_grid=grid,
            raster_options=RasterOptions(metal_smoothing="staircase"),
            sources=[source],
            monitors=monitors,
            boundaries=[
                bz.Periodic(axes=("x", "y")),
                bz.PML(
                    edges=("front", "back"),
                    thickness=200e-9,
                    formulation="cpml",
                    alpha_max=0,
                    kappa_max=2,
                ),
            ],
            time=np.arange(0, 300e-15 + dt / 2, dt),
        )
        point = MaterialGrid.from_raster_result(
            rasterize(
                scene,
                grid,
                options=RasterOptions(periodic_axes=("x", "y")),
                _staircase_materials=tuple(range(len(scene.materials))),
            ),
            dimensions=3,
        )
        sim = sim.updated_copy(
            material_grid=replace(sim.material_grid, yee_materials=point.yee_materials)
        )
        if args.meep_ownership and scene.objects:
            saved = np.load(
                ROOT
                / f"docs/reviews/cmos-parity-followup/meep/dx{args.dx:g}-length{args.length:g}/beamz_sites.npz"
            )
            yee = dict(sim.material_grid.yee_materials)
            support = dict(sim.material_grid.dispersion[0][1])
            for c in ("Ex", "Ey", "Ez"):
                coords = component_coordinates_rectilinear(c, grid)
                mask = (coords["z"] >= -160e-9) & (coords["z"] <= 110e-9)
                metal = saved[c].real < 0
                key = "eps_" + c[-1].lower()
                values = np.array(yee[key])
                values[mask] = np.where(metal, gold.epsilon_inf, saved[c].real)
                yee[key] = values
                weights = np.array(support[c])
                weights[mask] = metal
                support[c] = weights
            sim = sim.updated_copy(
                material_grid=replace(
                    sim.material_grid, yee_materials=yee, dispersion=((gold, support),)
                )
            )
        if args.coefficients_only:
            values = {}
            for c in ("Ex", "Ey", "Ez"):
                xyz = component_coordinates_rectilinear(c, grid)
                mask = (xyz["z"] >= -160e-9) & (xyz["z"] <= 110e-9)
                for a in "xyz":
                    values[c + "_" + a] = xyz[a][mask] if a == "z" else xyz[a]
                eps = np.asarray(
                    point.yee_materials["eps_" + c[-1].lower()], dtype=complex
                )
                for m, support in sim.material_grid.dispersion:
                    eps += (m.eps_model(f0) - m.epsilon_inf) * support[c]
                values[c] = eps[mask]
            np.savez(out / "coefficients.npz", **values)
            return
        print(
            json.dumps(
                dict(
                    grid=grid.shape,
                    dt=sim.dt,
                    device=str(jax.devices()),
                    objects=len(scene.objects),
                )
            ),
            flush=True,
        )
        result = sim.advance(backend="jax", progress=False, donate_state=True).results
        e = {}
        for n in ("r0", "r1", "i0", "i1"):
            w = np.asarray(result[n].integration_weights)
            values = fields(result[n])["Ey"]
            if not w.size:  # Uniform-grid flux uses implicit equal-area quadrature.
                w = np.full(values.shape[1], dx**2)
            e[n] = np.sum(values * w, axis=1) / w.sum()
        results.append((e, np.asarray(result["t"].flux), np.asarray(result["i0"].flux)))
        jax.clear_caches()

    def current(e, a, b):
        return -np.imag(e[a] * e[b].conj()) / (planes[b] - planes[a])

    ref, dev = results
    incident = -current(ref[0], "i0", "i1") + current(ref[0], "r0", "r1")
    reflection = current(dev[0], "r0", "r1") / incident
    transmission = dev[1] / ref[2]
    report = dict(
        dx_nm=args.dx,
        length_nm=args.length,
        device=str(jax.devices()),
        dt=dt,
        metal_model="Drude matched at 850 nm",
        epsilon_850=[epsilon.real, epsilon.imag],
        epsilon_inf=gold.epsilon_inf,
        gamma_rad_s=gamma,
        plasma_rad_s=plasma,
        all_interfaces_staircased=True,
        simplified_jax_constants=bool(jax.config.jax_use_simplified_jaxpr_constants),
        meep_ownership=args.meep_ownership,
        reflection_850=float(reflection[20]),
        transmission_850=float(transmission[20]),
        empty_reflection_850=float((current(ref[0], "r0", "r1") / incident)[20]),
        elapsed_s=time.monotonic() - start,
    )
    np.savez(
        out / "spectra.npz",
        frequency_hz=freq,
        reflection=reflection,
        transmission=transmission,
        incident=incident,
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
