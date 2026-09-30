"""Controlled three-material junction variants and native-Yee material loss.

No solver defaults change. The private monitor-compiler hook replaces diagnostic
point placeholders with direct gathers at metal-owned Yee sites. This avoids
interpolating E across metal boundaries when evaluating absorbed power.
"""

import argparse
import importlib
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import metal_benchmarks as benchmark
import numpy as np

import beamz as bz
from beamz.const import MU_0
from beamz.design.discretization import MaterialGrid
from beamz.design.raster import RasterOptions
from beamz.lattice import component_shapes

COMPONENTS = ("Ex", "Ey", "Ez")


def fractions(scene, grid):
    """Obtain painter-ordered volume fractions without the metal policy."""
    result = []
    for owner in range(len(scene.materials)):
        indicators = tuple(
            bz.Material(1, conductivity=float(i == owner))
            for i in range(len(scene.materials))
        )
        raster = replace(scene, materials=indicators).rasterize(
            grid,
            options=RasterOptions(smoothing="volume", periodic_axes=("x", "y")),
        )
        mg = MaterialGrid.from_raster_result(raster, dimensions=3)
        result.append(
            {
                c: np.asarray(mg.yee_materials["sig_" + c[-1].lower()])
                for c in COMPONENTS
            }
        )
    return result


def junction_variant(scene, grid, mg, variant):
    if len(scene.materials) == 1:
        return mg, {}
    f = fractions(scene, grid)
    metal = mg.dispersion[0][1]
    coefficients = dict(mg.yee_materials)
    audit = {}
    for c in COMPONENTS:
        nonmetal = np.stack([f[i][c] for i in (0, 2, 3)])
        total = nonmetal.sum(axis=0)
        mask = (
            (metal[c] < 1e-7)
            & (f[1][c] > 1e-12)
            & (np.count_nonzero(nonmetal > 1e-12, axis=0) >= 2)
        )
        eps = np.array([1.0, 1.892, 2.25])[:, None, None, None]
        weights = nonmetal / np.maximum(total, 1e-30)
        effective = (
            1 / np.maximum((weights / eps).sum(axis=0), 1e-30)
            if c == "Ez" and variant == "polarized"
            else (weights * eps).sum(axis=0)
        )
        key = "eps_" + c[-1].lower()
        old = np.asarray(coefficients[key])
        audit[c] = dict(
            junction_sites=int(mask.sum()),
            changed_sites=int(np.count_nonzero(mask & (abs(old - effective) > 1e-12))),
            max_epsilon_change=float(np.max(np.where(mask, abs(old - effective), 0))),
        )
        if variant != "baseline":
            coefficients[key] = np.where(mask, effective, old)
    changed = replace(mg, yee_materials=coefficients)
    # Pole ownership is deliberately unchanged, including periodic endpoint faces.
    for c in COMPONENTS:
        np.testing.assert_array_equal(changed.dispersion[0][1][c], metal[c])
    return changed, audit


def support_volumes(grid, c):
    """Dual transverse widths and primal longitudinal widths, SI m³.

    Endpoint halves prevent double counting duplicated periodic boundary sites.
    """
    widths = []
    for axis, edges in zip("xyz", grid.edges, strict=True):
        d = np.diff(edges)
        widths.append(
            d if axis == c[-1].lower() else np.r_[d[0], d[:-1] + d[1:], d[-1]] / 2
        )
    return (
        widths[2][:, None, None] * widths[1][None, :, None] * widths[0][None, None, :]
    )


def native_monitor_compiler(original, indices):
    def compile_native(*args, **kwargs):
        specs, records = original(*args, **kwargs)
        output = []
        offset = 0
        for spec in specs:
            if spec.name.startswith("loss_"):
                c = spec.name[5:]
                idx = jnp.asarray(indices[c][:, None], dtype=jnp.int32)
                n = len(indices[c])
                zeros = jnp.zeros((n, 1), dtype=jnp.int32)
                weights = jnp.ones((n, 1), dtype=jnp.float32)
                spec = replace(
                    spec,
                    dft_point_count=n,
                    dft_flat_idx=tuple(
                        idx if name == c else zeros
                        for name in (*COMPONENTS, "Hx", "Hy", "Hz")
                    ),
                    dft_weights=tuple(
                        weights if name == c else weights * 0
                        for name in (*COMPONENTS, "Hx", "Hy", "Hz")
                    ),
                    sample_region=None,
                    integration_weights=jnp.zeros(n),
                )
            spec = replace(spec, dft_value_offset=offset)
            offset += 6 * spec.freq_count * spec.dft_point_count
            output.append(spec)
        return tuple(output), records

    return compile_native


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=("baseline", "volume", "polarized"), default="baseline"
    )
    parser.add_argument("--mesh", choices=("reference", "refined"), default="reference")
    parser.add_argument("--dx", type=float, default=5)
    parser.add_argument("--length", type=float, default=132.3076923076923)
    args = parser.parse_args()
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("Run this comparison on the local GPU")
    benchmark.OUT /= f"junction-loss/{args.variant}"
    incident = None

    def execute(
        scene, grid, sources, monitors, boundaries, runtime_fs, courant=0.8, **kwargs
    ):
        nonlocal incident
        dt = grid.cfl_time_step(courant)
        freq = bz.LIGHT_SPEED / 850e-9
        sim = bz.Simulation(
            scene=scene,
            raster_grid=grid,
            raster_options=RasterOptions(reference_frequency=freq),
            sources=sources,
            monitors=monitors,
            boundaries=boundaries,
            time=np.arange(0, runtime_fs * 1e-15 + dt / 2, dt),
            normalize_source=None,
        )
        mg, audit = junction_variant(scene, grid, sim.material_grid, args.variant)
        indices = {}
        volume = {}
        extra = []
        if mg.dispersion:
            for c in COMPONENTS:
                mask = np.asarray(mg.dispersion[0][1][c]) > 1e-7
                indices[c] = np.flatnonzero(mask)
                volume[c] = support_volumes(grid, c).ravel()[indices[c]]
                extra.append(
                    bz.FieldMonitor(
                        center=(0, 0, 0),
                        size=(0, 0, 0),
                        freqs=[freq],
                        fields=(c,),
                        name="loss_" + c,
                        interval=10,
                    )
                )
        metal_count = {c: len(i) for c, i in indices.items()}
        plane_blocks = {}
        plane_weights = {}
        if mg.dispersion:
            shapes = component_shapes(mg.shape)
            z = grid.edges[2]
            heights = {m.name: m.center[2] for m in monitors}
            plane_indices = {
                name: int(np.argmin(abs(z - heights[name]))) for name in ("input0", "t")
            }
            for c in ("Ex", "Ey", "Hx", "Hy"):
                blocks = list(indices.get(c, np.empty(0, dtype=int)))
                plane_blocks[c] = {}
                for name, zi in plane_indices.items():
                    count = int(np.prod(shapes[c][1:]))
                    for side in range(2 if c.startswith("E") else 1):
                        start = len(blocks)
                        blocks.extend(((zi + side) * count + np.arange(count)).tolist())
                        plane_blocks[c][(name, side)] = slice(start, len(blocks))
                    if c.startswith("E"):
                        dual_z = (z[zi + 1] - z[zi - 1]) / 2
                        plane_weights[c] = support_volumes(grid, c)[zi].ravel() / dual_z
                indices[c] = np.asarray(blocks, dtype=np.int64)
                if c.startswith("H"):
                    extra.append(
                        bz.FieldMonitor(
                            center=(0, 0, 0),
                            size=(0, 0, 0),
                            freqs=[freq],
                            fields=(c,),
                            name="loss_" + c,
                            interval=10,
                        )
                    )
        sim = sim.updated_copy(material_grid=mg, monitors=[*monitors, *extra])
        compiler = importlib.import_module("beamz.simulation.compile")
        with patch.object(
            compiler,
            "compile_monitor_specs",
            native_monitor_compiler(compiler.compile_monitor_specs, indices),
        ):
            first = sim.advance(
                num_steps=3 * sim.num_steps // 4,
                backend="jax",
                progress=False,
                donate_state=True,
            )
            result = sim.advance(
                state=first.state, backend="jax", progress=False, donate_state=True
            ).results

        def current(lo, hi):
            values = []
            for name in (lo, hi):
                m = result[name]
                w = np.asarray(m.integration_weights)
                values.append(
                    np.sum(m.get_dft_component("Ey") * w, axis=1)[0] / w.sum()
                )
            heights = {m.name: m.center[2] for m in monitors}
            return float(
                -np.imag(values[0] * values[1].conj()) / (heights[hi] - heights[lo])
            )

        if not mg.dispersion:
            incident = -current("input0", "input1") + current("r", "r2")
        meta = dict(dt=dt, grid=list(grid.shape), steps=sim.num_steps, junction=audit)
        if mg.dispersion:
            frequency_warped = np.tan(np.pi * freq * dt) / (np.pi * dt)
            eps_imag = float(scene.materials[1].eps_model(frequency_warped).imag)
            omega_discrete = 2 * np.sin(np.pi * freq * dt) / dt
            area = np.ptp(grid.edges[0]) * np.ptp(grid.edges[1])
            losses = {}
            for c in COMPONENTS:
                e = np.asarray(result["loss_" + c].get_dft_component(c))[
                    0, : metal_count[c]
                ]
                losses[c] = float(
                    (omega_discrete / bz.LIGHT_SPEED) ** 2
                    * eps_imag
                    * np.sum(volume[c] * abs(e) ** 2)
                    / (incident * area)
                )
            np.savez_compressed(
                benchmark.OUT
                / f"native_fields-{args.mesh}-{args.dx:g}-{args.length:g}.npz",
                **{
                    c: np.asarray(result["loss_" + c].get_dft_component(c))[0]
                    for c in indices
                },
                **{"idx_" + c: indices[c] for c in indices},
                **{"vol_" + c: volume[c] for c in COMPONENTS},
                **{a: e for a, e in zip("xyz", grid.edges, strict=True)},
                incident=incident,
                dt=dt,
            )
            absorption = sum(losses.values())

            def native_flux(name):
                power = 0.0
                for e_name, h_name, sign in (("Ex", "Hy", 1), ("Ey", "Hx", -1)):
                    e = np.asarray(result["loss_" + e_name].get_dft_component(e_name))[
                        0
                    ]
                    h = np.asarray(result["loss_" + h_name].get_dft_component(h_name))[
                        0
                    ]
                    lo, hi = (e[plane_blocks[e_name][(name, side)]] for side in (0, 1))
                    magnetic = h[plane_blocks[h_name][(name, 0)]] * np.exp(
                        -1j * np.pi * freq * dt
                    )
                    power += (
                        0.5
                        * sign
                        * np.sum(
                            plane_weights[e_name]
                            * np.real((lo + hi) / 2 * magnetic.conj())
                        )
                    )
                return float(power)

            native_incident = incident * area / (2 * MU_0 * omega_discrete)
            net_in = -native_flux("input0") / native_incident
            native_transmission = -native_flux("t") / native_incident
            reflection = current("r", "r2") / incident
            transmission = float(result["t"].flux[0])
            # Transmission is tiny here; retain conventional flux explicitly as secondary.
            transmission /= -incident * area / (2 * MU_0 * omega_discrete)
            drift = {}
            for c in COMPONENTS:
                old, new = first.results["loss_" + c], result["loss_" + c]
                a = old.get_dft_component(c) * old.dft_weight_sum[:, None]
                b = new.get_dft_component(c) * new.dft_weight_sum[:, None]
                drift[c] = float(np.max(abs(a - b)) / max(float(np.max(abs(b))), 1e-30))
            meta.update(
                absorption=absorption,
                absorption_components=losses,
                reflection=reflection,
                transmission=transmission,
                energy_residual=1 - reflection - transmission - absorption,
                native_net_entering=net_in,
                native_transmission=native_transmission,
                native_energy_residual=net_in - native_transmission - absorption,
                native_electric_checkpoint_drift=drift,
                epsilon_imag_discrete=eps_imag,
            )
        print(json.dumps(meta), flush=True)
        return result, meta

    benchmark.execute = execute
    benchmark.reflector(
        SimpleNamespace(
            dx=args.dx,
            runtime=300,
            mesh=args.mesh,
            gold_model="snapshot",
            lengths=[args.length],
        )
    )


if __name__ == "__main__":
    main()
