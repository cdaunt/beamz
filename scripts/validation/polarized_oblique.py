"""Periodic oblique TE/TM half-space check of dispersive interface averaging.

The x period is one incident transverse wavelength: no Bloch phase is needed
at the measured 550 nm frequency. Sample transmitted flux inside the medium.
"""

import argparse
import json
import sys
from pathlib import Path

import jax
import numpy as np

import beamz as bz
from beamz.design.raster import RasterOptions

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
from cmos_rgb_sensor import load_materials  # noqa: E402


def run(dx_nm, polarization, shift, smoothing, medium=None):
    dx = dx_nm * 1e-9
    wavelength = 550e-9
    frequency = bz.LIGHT_SPEED / wavelength
    size = (1.1e-6, 4 * dx, 2.4e-6)
    surface = shift * dx
    depth = 4 * dx
    design = bz.Design(background=bz.Material(1))
    if medium is not None:
        design += bz.Box(
            center=(0, 0, (surface - size[2] / 2) / 2),
            size=(*size[:2], surface + size[2] / 2),
            material=medium,
        )
    sim = bz.Simulation(
        design=design,
        size=size,
        raster_options=RasterOptions(smoothing=smoothing),
        sources=[
            bz.GaussianBeamSource(
                center=(0, 0, 0.65e-6),
                size=(*size[:2], 0),
                direction="-z",
                source_time=bz.GaussianPulse(frequency, 0.12 * frequency),
                angle_theta=np.pi / 6,
                angle_phi=0,
                pol_angle=0 if polarization == "TM" else np.pi / 2,
                waist_radius=np.inf,
            )
        ],
        monitors=[
            bz.FluxMonitor(
                center=(0, 0, z), size=(*size[:2], 0), freqs=[frequency], name=name
            )
            for name, z in (("input", 0.35e-6), ("output", surface - depth))
        ],
        boundaries=[
            bz.Periodic(axes=("x", "y")),
            bz.PML(edges=("front", "back"), thickness=0.35e-6, formulation="cpml"),
        ],
        grid_spec=bz.GridSpec.uniform(dx, courant=0.7),
        run_time=180e-15,
    )
    results = sim.run(backend="jax", progress=False)
    return {
        name: float(np.asarray(results[name].flux)[0]) for name in ("input", "output")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx-nm", type=float, default=10)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--material", choices=("silicon", "metal", "sin", "silica"), default="silicon"
    )
    parser.add_argument("--shifts", type=float, nargs="+", default=[0, 0.25, 0.5])
    args = parser.parse_args()
    if not args.allow_cpu and not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("This validation requires the local GPU")
    output = ROOT / "docs/reviews/cmos-polarized/oblique"
    output.mkdir(parents=True, exist_ok=True)
    medium = load_materials()[args.material]
    epsilon = complex(medium.eps_model(bz.LIGHT_SPEED / 550e-9))
    kz = np.sqrt(epsilon - 0.25)
    records = []
    for pol in ("TE", "TM"):
        y1 = np.cos(np.pi / 6) if pol == "TE" else 1 / np.cos(np.pi / 6)
        y2 = kz if pol == "TE" else epsilon / kz
        exact_r = abs((y1 - y2) / (y1 + y2)) ** 2
        exact_t = float(
            y2.real
            / y1
            * abs(2 * y1 / (y1 + y2)) ** 2
            * np.exp(-4 * np.pi * kz.imag * (4 * args.dx_nm * 1e-9) / 550e-9)
        )
        for shift in args.shifts:
            ref = run(args.dx_nm, pol, shift, "volume")
            for smoothing in ("volume", "farjadpour_diagonal"):
                dev = run(args.dx_nm, pol, shift, smoothing, medium)
                r = 1 - dev["input"] / ref["input"]
                t = dev["output"] / ref["output"]
                record = dict(
                    device=str(jax.devices()),
                    boundary="cpml",
                    material=args.material,
                    dx_nm=args.dx_nm,
                    polarization=pol,
                    shift=shift,
                    smoothing=smoothing,
                    R=r,
                    T=t,
                    R_exact=exact_r,
                    T_exact=exact_t,
                    R_error=r - exact_r,
                    T_error=t - exact_t,
                    reference_flux_ratio=ref["output"] / ref["input"],
                )
                records.append(record)
                print(json.dumps(record), flush=True)
                (output / f"{args.material}-{args.dx_nm:g}nm.json").write_text(
                    json.dumps(records, indent=2) + "\n"
                )
            jax.clear_caches()


if __name__ == "__main__":
    main()
