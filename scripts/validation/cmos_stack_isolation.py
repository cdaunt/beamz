"""Reference-depth planar CMOS stacks compared with complex transfer matrices."""

import argparse
import json
import sys
from pathlib import Path

import jax
import numpy as np

import beamz as bz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
from cmos_reference_mesh import make_reference_mesh_sim  # noqa: E402
from cmos_rgb_sensor import Lx, Ly, Lz, load_materials, t_ar, t_silicon  # noqa: E402


def stack_solution(channel, wl):
    m = load_materials()
    s = json.loads(
        (ROOT / "examples/data/cmos_rgb/tidy3d_reference_simulation.json").read_text()
    )
    geom = {o["name"]: o["geometry"] for o in s["structures"]}
    top = (
        geom["silica layer"]["center"][2] + geom["silica layer"]["size"][2] / 2
    ) * 1e-6
    si = -Lz / 2 + t_silicon
    layers = [
        (m["silica"], 0.2e-6),
        (m[channel], 1e-6),
        (m["silica"], top - si - t_ar - 1.2e-6),
        (m["sin"], t_ar),
    ]
    matrix = np.broadcast_to(np.eye(2, dtype=complex), (len(wl), 2, 2)).copy()
    for mat, d in layers:
        n = np.sqrt(mat.eps_model(bz.LIGHT_SPEED / wl))
        phase = 2 * np.pi * n * d / wl
        a = np.zeros_like(matrix)
        a[:, 0, 0] = a[:, 1, 1] = np.cos(phase)
        a[:, 0, 1] = -1j * np.sin(phase) / n
        a[:, 1, 0] = -1j * n * np.sin(phase)
        matrix = matrix @ a
    ns = np.sqrt(m["silicon"].eps_model(bz.LIGHT_SPEED / wl))
    b = matrix[:, 0, 0] + matrix[:, 0, 1] * ns
    c = matrix[:, 1, 0] + matrix[:, 1, 1] * ns
    t = 2 / (b + c)
    r = (b - c) / (b + c)
    return top, si, abs(r) ** 2, ns.real * abs(t) ** 2, t, ns


def run(channel, refine, boundary, wl, courant=0.7):
    top, si, *_ = stack_solution("red" if channel == "reference" else channel, wl)
    freq = bz.LIGHT_SPEED / wl
    positions = {"input": 2.5e-6, "top": top, "interface": si, "inside": si - 50e-9}
    monitors = [
        bz.FluxMonitor(center=(0, 0, z), size=(Lx, Ly, 0), freqs=freq, name=name)
        for name, z in positions.items()
    ]
    monitors.append(
        bz.FluxMonitor(
            center=(0, 0, si),
            size=(Lx, Ly, 0),
            freqs=freq,
            name="one_sided",
            colocation_side="plus",
        )
    )
    sim = make_reference_mesh_sim(
        lateral_spacing=750e-9,
        depth_refinement=refine,
        boundary=boundary,
        courant=courant,
        reference=channel == "reference",
        stage="full" if channel == "reference" else f"planar_{channel}",
        sources=[
            bz.PlaneWaveSource(
                center=(0, 0, 2.8047480102767586e-6),
                size=(Lx, Ly, 0),
                direction="-z",
                source_time=bz.GaussianPulse(5.75e14, 2e14, offset=5),
            )
        ],
        monitors=monitors,
        run_time=300e-15,
    )
    result = sim.run(backend="jax", progress=False)
    data = {name: np.asarray(result[name].flux) for name in (*positions, "one_sided")}
    data.update(
        E_top=result["top"].get_dft_component("Ex").mean(axis=1),
        E_interface=result["interface"].get_dft_component("Ex").mean(axis=1),
    )
    return data, dict(grid=list(sim.grid.shape), dt=sim.dt, steps=sim.num_steps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--refinements", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--courant", type=float, default=0.7)
    p.add_argument(
        "--channels",
        nargs="+",
        choices=["red", "green", "blue"],
        default=["red", "green", "blue"],
    )
    p.add_argument("--boundary", choices=["absorber", "cpml"], default="absorber")
    args = p.parse_args()
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("GPU required")
    label = args.boundary
    if args.courant != 0.7:
        label += f"-courant-{args.courant:g}"
    out = ROOT / "docs/reviews/cmos-isolation/planar" / label
    out.mkdir(parents=True, exist_ok=True)
    wl = np.linspace(400e-9, 700e-9, 61)
    records = (
        json.loads((out / "report.json").read_text())["runs"]
        if (out / "report.json").exists()
        else []
    )
    records = [
        r
        for r in records
        if r["depth_refinement"] not in args.refinements
        or r["channel"] not in args.channels
    ]
    for refine in args.refinements:
        ref, meta = run("reference", refine, args.boundary, wl, args.courant)
        for channel in args.channels:
            dev, _ = run(channel, refine, args.boundary, wl, args.courant)
            _, _, R, T, t, n = stack_solution(channel, wl)
            measured_R = 1 - dev["input"] / ref["input"]
            Ts = {
                k: dev[k] / ref["input"] for k in ("interface", "one_sided", "inside")
            }
            transmitted = dev["E_interface"] / ref["E_top"]
            expected_inside = T * np.exp(-4 * np.pi * n.imag * 50e-9 / wl)
            row = dict(
                channel=channel,
                clock_policy="integer_step",
                courant=args.courant,
                depth_refinement=refine,
                boundary=args.boundary,
                **meta,
                R_error_pp=float(100 * np.max(abs(measured_R - R))),
                T_error_pp={
                    k: float(
                        100 * np.max(abs(v - (expected_inside if k == "inside" else T)))
                    )
                    for k, v in Ts.items()
                },
                t_complex_max_error=float(np.max(abs(transmitted - t))),
                min_absorption=float(np.min(1 - measured_R - Ts["one_sided"])),
            )
            print(json.dumps(row), flush=True)
            records.append(row)
            np.savez(
                out / f"{channel}-refine{refine}.npz",
                wavelength_nm=wl * 1e9,
                R=measured_R,
                R_exact=R,
                T_exact=T,
                T_inside_exact=expected_inside,
                t_exact=t,
                t=transmitted,
                **Ts,
            )
            (out / "report.json").write_text(
                json.dumps(dict(device=str(jax.devices()), runs=records), indent=2)
                + "\n"
            )
            jax.clear_caches()


if __name__ == "__main__":
    main()
