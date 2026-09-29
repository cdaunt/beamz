"""Broadband analytical check of the imported 100 nm aluminum shield material."""

import json
import sys
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
from cmos_planar_checks import analytic_stack, run  # noqa: E402
from cmos_rgb_sensor import load_materials  # noqa: E402


def main():
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("Local GPU required")
    out = ROOT / "docs/reviews/cmos-isolation/metal-film"
    out.mkdir(parents=True, exist_ok=True)
    wl = np.linspace(400e-9, 700e-9, 61)
    layers = [(load_materials()["metal"], 100e-9)]
    exact_r, exact_t = analytic_stack(layers, wl)
    rows = []
    for dx_nm in (10, 5, 2.5):
        ref = run([(None, 100e-9)], dx_nm * 1e-9, wl)
        dev = run(layers, dx_nm * 1e-9, wl)
        r = 1 - dev["input"] / ref["input"]
        t = dev["output"] / ref["output"]
        rows.append(
            dict(
                dx_nm=dx_nm,
                R_error_pp=float(100 * max(abs(r - exact_r))),
                T_error_pp=float(100 * max(abs(t - exact_t))),
                min_absorption=float(min(1 - r - t)),
            )
        )
        np.savez(
            out / f"{dx_nm:g}nm.npz",
            wavelength_nm=wl * 1e9,
            R=r,
            T=t,
            R_exact=exact_r,
            T_exact=exact_t,
        )
        print(rows[-1], flush=True)
        jax.clear_caches()
    report = dict(
        device=str(jax.devices()),
        clock_policy="integer_step",
        film_thickness_nm=100,
        runs=rows,
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    assert rows[-1]["R_error_pp"] < 0.5 and rows[-1]["T_error_pp"] < 0.5
    assert all(r["min_absorption"] > -0.005 for r in rows)


if __name__ == "__main__":
    main()
