"""Isolate interface flux sampling using independent native H-plane probes."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

import beamz as bz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
from cmos_rgb_sensor import load_materials  # noqa: E402


def extrapolation_weights(points, target):
    points = np.asarray(points, dtype=float)
    weights = np.ones(len(points))
    for j, x in enumerate(points):
        for k, y in enumerate(points):
            if j != k:
                weights[j] *= (target - y) / (x - y)
    return weights


def run(dx_nm, shift, reference=False):
    dx = dx_nm * 1e-9
    surface = shift * dx
    size = (4 * dx, 4 * dx, 3.2e-6)
    design = bz.Design(background=bz.Material(1))
    if not reference:
        design += bz.Box(
            center=(0, 0, (surface - size[2] / 2) / 2),
            size=(*size[:2], surface + size[2] / 2),
            material=load_materials()["silicon"],
        )
    wl = np.linspace(400e-9, 700e-9, 31)
    freq = bz.LIGHT_SPEED / wl
    positions = {"input": 0.5e-6, "interface": surface}
    for side, sign in (("minus", -1), ("plus", 1)):
        nodes = (np.arange(-6, 7) + 0.5) * dx
        nodes = nodes[sign * (nodes - surface) > dx * 1e-9]
        nodes = sorted(nodes, key=lambda p: abs(p - surface))[:3]
        positions.update({f"{side}{i}": float(z) for i, z in enumerate(nodes)})
    monitors = [
        bz.FluxMonitor(center=(0, 0, z), size=(*size[:2], 0), freqs=freq, name=name)
        for name, z in positions.items()
    ]
    for side in ("minus", "plus"):
        monitors.append(
            bz.FluxMonitor(
                center=(0, 0, surface),
                size=(*size[:2], 0),
                freqs=freq,
                name=f"compiled_{side}",
                colocation_side=side,
            )
        )
    sim = bz.Simulation(
        design=design,
        size=size,
        sources=[
            bz.PlaneWaveSource(
                center=(0, 0, 0.85e-6),
                size=(*size[:2], 0),
                direction="-z",
                source_time=bz.GaussianPulse(5.75e14, 2e14),
            )
        ],
        monitors=monitors,
        boundaries=[
            bz.Periodic(axes=("x", "y")),
            bz.PML(edges=("front", "back"), thickness=0.5e-6, formulation="cpml"),
        ],
        grid_spec=bz.GridSpec.uniform(dx, courant=0.7),
        run_time=130e-15,
    )
    result = sim.run(backend="jax", progress=False)
    monitor = result["interface"]
    spectra = {
        "input": np.asarray(result["input"].flux),
        "linear": np.asarray(monitor.flux),
    }
    spectra.update(
        {
            f"compiled_{side}": np.asarray(result[f"compiled_{side}"].flux)
            for side in ("minus", "plus")
        }
    )
    raw = {
        f"{name}_{c}": np.asarray(result[name].dft_fields[c])
        for name in positions
        for c in ("Ex", "Ey", "Hx", "Hy")
    }
    for side in ("minus", "plus"):
        for order in (1, 2):
            names = [f"{side}{j}" for j in range(order + 1)]
            weights = extrapolation_weights([positions[n] for n in names], surface)
            fields = dict(monitor.dft_fields)
            for component in ("Hx", "Hy"):
                fields[component] = sum(
                    w * result[n].dft_fields[component]
                    for w, n in zip(weights, names, strict=True)
                )
            spectra[f"{side}{order}"] = np.asarray(
                replace(monitor, dft_fields=fields).flux
            )
        # Spatial summation before/after a long float32 DFT differs by roundoff.
        if shift != 0.5:
            np.testing.assert_allclose(
                spectra[f"compiled_{side}"], spectra[f"{side}2"], rtol=1e-4, atol=1e-8
            )
    raw.update(
        positions=np.array(list(positions.values())),
        names=np.array(list(positions)),
        dt=sim.dt,
        dft_weight_sum=monitor.dft_weight_sum,
        wavelength_nm=wl * 1e9,
    )
    return spectra, raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx-nm", type=float, nargs="+", default=[10, 5, 2.5])
    parser.add_argument("--shifts", type=float, nargs="+", default=[0, 0.25, 0.5])
    args = parser.parse_args()
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("Local GPU required")
    out = ROOT / "docs/reviews/cmos-isolation/reconstruction"
    out.mkdir(parents=True, exist_ok=True)
    wl = np.linspace(400e-9, 700e-9, 31)
    n = np.sqrt(load_materials()["silicon"].eps_model(bz.LIGHT_SPEED / wl))
    exact_T = 4 * n.real / abs(1 + n) ** 2
    exact_R = abs((1 - n) / (1 + n)) ** 2
    records = []
    for dx in args.dx_nm:
        reference, _ = run(dx, 0, True)
        for shift in args.shifts:
            spectra, raw = run(dx, shift)
            row = dict(
                dx_nm=dx,
                shift=shift,
                device=str(jax.devices()),
                boundary="cpml",
                clock_policy="integer_step",
            )
            normalized = {k: v / reference["input"] for k, v in spectra.items()}
            measured_R = 1 - normalized.pop("input")
            row["R_max_error_pp"] = float(100 * np.max(abs(measured_R - exact_R)))
            row["T_max_error_pp"] = {
                k: float(100 * np.max(abs(v - exact_T))) for k, v in normalized.items()
            }
            print(json.dumps(row), flush=True)
            records.append(row)
            np.savez(
                out / f"{dx:g}nm-shift{shift:g}.npz",
                **raw,
                **normalized,
                R=measured_R,
                T_exact=exact_T,
                R_exact=exact_R,
            )
            (out / "report.json").write_text(json.dumps(records, indent=2) + "\n")
            jax.clear_caches()


if __name__ == "__main__":
    main()
