"""Independent oblique Fresnel controls for the imported CMOS dielectrics.

Each measured frequency has exactly one transverse wavelength in the periodic
cell at 30-degree incidence. Reference-normalized monitors are in homogeneous
regions, four cells inside the medium, away from the smoothed interface.
"""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from polarized_oblique import load_materials, run

import beamz as bz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx-nm", type=float, nargs="+", default=[10, 5])
    parser.add_argument("--wavelength-nm", type=float, nargs="+", default=[450, 650])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/reviews/cmos-isolation/periodic-interfaces/fresnel.json"),
    )
    args = parser.parse_args()
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("Local GPU required")
    records = []
    materials = load_materials()
    for dx in args.dx_nm:
        for wavelength_nm in args.wavelength_nm:
            wavelength = wavelength_nm * 1e-9
            for polarization in ("TE", "TM"):
                reference = run(
                    dx,
                    polarization,
                    0.25,
                    "farjadpour_diagonal",
                    wavelength_nm=wavelength_nm,
                )
                for name in ("silica", "sin"):
                    medium = materials[name]
                    epsilon = complex(medium.eps_model(bz.LIGHT_SPEED / wavelength))
                    kz = np.sqrt(epsilon - 0.25)
                    y1 = (
                        np.cos(np.pi / 6)
                        if polarization == "TE"
                        else 1 / np.cos(np.pi / 6)
                    )
                    y2 = kz if polarization == "TE" else epsilon / kz
                    exact_r = abs((y1 - y2) / (y1 + y2)) ** 2
                    exact_t = float(
                        y2.real
                        / y1
                        * abs(2 * y1 / (y1 + y2)) ** 2
                        * np.exp(-4 * np.pi * kz.imag * 4 * dx * 1e-9 / wavelength)
                    )
                    device = run(
                        dx,
                        polarization,
                        0.25,
                        "farjadpour_diagonal",
                        medium,
                        wavelength_nm=wavelength_nm,
                    )
                    measured_r = 1 - device["input"] / reference["input"]
                    measured_t = device["output"] / reference["output"]
                    assert np.isfinite([measured_r, measured_t]).all()
                    record = dict(
                        dx_nm=dx,
                        wavelength_nm=wavelength_nm,
                        polarization=polarization,
                        material=name,
                        shift_cells=0.25,
                        R=measured_r,
                        T=measured_t,
                        R_exact=float(exact_r),
                        T_exact=exact_t,
                        R_error_pp=float(100 * (measured_r - exact_r)),
                        T_error_pp=float(100 * (measured_t - exact_t)),
                        reference_flux_ratio=reference["output"] / reference["input"],
                    )
                    records.append(record)
                    print(json.dumps(record), flush=True)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(
                        json.dumps(
                            dict(
                                device=str(jax.devices()),
                                angle_degrees=30,
                                smoothing="farjadpour_diagonal",
                                runs=records,
                            ),
                            indent=2,
                        )
                        + "\n"
                    )
                    jax.clear_caches()


if __name__ == "__main__":
    main()
