"""Paired linear/one-sided detectors and staged geometry on the fixed CMOS mesh."""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
from cmos_rgb_sensor import make_sim, pixel_signs  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=["bare_stack", "no_lenses", "full"], required=True
    )
    args = parser.parse_args()
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("Local GPU required")
    out = ROOT / "docs/reviews/cmos-isolation/components" / args.stage
    out.mkdir(parents=True, exist_ok=True)
    old = ROOT / "docs/reviews/cmos-polarized/reference-z-20nm-absorber"
    sim = make_sim(
        dx=20e-9,
        run_time=510e-15,
        num_freqs=200,
        fields=False,
        reference_z=True,
        stage=args.stage,
        colocation_side="plus",
        compare_colocation=True,
    )
    prior = json.loads((old / "reference_report.json").read_text())
    # Reuse only the identical empty-cell excitation and numerical domain.
    assert sim.dt == prior["dt"] and sim.num_steps == prior["steps"]
    assert sim.sources[0].source_time.offset == prior["source_offset"]
    grid = np.load(old / "reference_grid.npz")
    for axis, edges in zip("xyz", sim.grid.edges, strict=True):
        np.testing.assert_array_equal(edges, grid[axis])
    ref = np.load(old / "reference_spectra.npz")
    np.testing.assert_allclose(
        ref["wavelength_nm"], 299792458 / np.asarray(sim.monitors[0].freqs) * 1e9
    )
    checkpoint = int(300e-15 / sim.dt)
    print(
        "Running",
        args.stage,
        "grid",
        sim.grid.shape,
        "steps",
        sim.num_steps,
        flush=True,
    )
    start = time.perf_counter()
    first = sim.advance(num_steps=checkpoint, backend="jax", performance=False)
    print("Checkpoint", time.perf_counter() - start, flush=True)
    second = sim.advance(
        state=first.state,
        num_steps=sim.num_steps - checkpoint,
        backend="jax",
        performance=False,
        donate_state=True,
    )
    elapsed = time.perf_counter() - start
    assert all(
        np.isfinite(np.asarray(getattr(second.state, c))).all()
        for c in ("ex", "ey", "ez", "hx", "hy", "hz")
    )
    assert all(np.isfinite(np.asarray(q)).all() for q in second.state.polarization)
    raw = {p: -np.asarray(second.results[f"flux_si_{p}"].flux) for p in pixel_signs}
    linear = {
        p: -np.asarray(second.results[f"flux_si_{p}_linear"].flux) for p in pixel_signs
    }
    earlier = {p: -np.asarray(first.results[f"flux_si_{p}"].flux) for p in pixel_signs}
    wl = ref["wavelength_nm"]
    modal_path = ROOT / "docs/reviews/cmos-isolation/incident/spectra.npz"
    modal = np.load(modal_path)
    np.testing.assert_allclose(modal["wavelength_nm"], wl, rtol=1e-12)
    incident = modal["modal_power"]
    order = np.argsort(wl)
    report = dict(
        stage=args.stage,
        clock_policy="integer_step",
        incident_normalization="forward plane-wave modal power from native E samples",
        modal_reference_sha256=hashlib.sha256(modal_path.read_bytes()).hexdigest(),
        device=str(jax.devices()),
        seconds=elapsed,
        grid=list(sim.grid.shape),
        dt=sim.dt,
        steps=sim.num_steps,
        checkpoint_fs=checkpoint * sim.dt * 1e15,
        colocation_side="plus",
        reference=str(old.relative_to(ROOT)),
        reference_sha256=hashlib.sha256(
            (old / "reference_spectra.npz").read_bytes()
        ).hexdigest(),
        snapshot_sha256=hashlib.sha256(
            (
                ROOT / "examples/data/cmos_rgb/tidy3d_reference_simulation.json"
            ).read_bytes()
        ).hexdigest(),
        central_efficiency_percent={},
        linear_central_percent={},
        sampling_max_change_pp={},
        temporal_max_change_pp={},
    )
    channels = {}
    original = {}
    for c, w in [("red", 650), ("green", 550), ("blue", 450)]:

        def combine(d, channel=c):
            return d["green1"] + d["green2"] if channel == "green" else d[channel]

        channels[c] = combine(raw) / incident
        original[c] = combine(linear) / incident
        report["central_efficiency_percent"][c] = float(
            100 * np.interp(w, wl[order], channels[c][order])
        )
        report["linear_central_percent"][c] = float(
            100 * np.interp(w, wl[order], original[c][order])
        )
        report["sampling_max_change_pp"][c] = float(
            100 * np.max(abs(channels[c] - original[c]))
        )
        report["temporal_max_change_pp"][c] = float(
            100 * np.max(abs(channels[c] - combine(earlier) / incident))
        )
    np.savez(out / "sensor_spectra.npz", wavelength_nm=wl, **raw)
    np.savez(out / "linear_spectra.npz", wavelength_nm=wl, **linear)
    np.savez(out / "checkpoint_spectra.npz", wavelength_nm=wl, **earlier)
    np.savez(out / "calibrated_efficiency.npz", wavelength_nm=wl, **channels)
    np.savez(out / "linear_efficiency.npz", wavelength_nm=wl, **original)
    np.savez(out / "grid.npz", **dict(zip("xyz", sim.grid.edges, strict=True)))
    shutil.copyfile(modal_path, out / "incident_calibration.npz")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
