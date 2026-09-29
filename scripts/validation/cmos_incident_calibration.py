"""Measure incident modal power without Yee-plane interpolation attenuation."""

import argparse
import json
import sys
from pathlib import Path

import jax
import numpy as np

import beamz as bz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
from cmos_reference_mesh import (  # noqa: E402
    decompose_normal_air_wave,
    make_reference_mesh_sim,
    reference_grid,
)
from cmos_rgb_sensor import Lx, Ly, Lz, d_absorber_space  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary", choices=["absorber", "cpml"], default="absorber")
    parser.add_argument("--optical-dz-nm", type=float)
    args = parser.parse_args()
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("GPU required")
    frequencies = np.linspace(bz.LIGHT_SPEED / 700e-9, bz.LIGHT_SPEED / 400e-9, 200)
    f0 = np.mean([frequencies.min(), frequencies.max()])
    max_dz = None if args.optical_dz_nm is None else args.optical_dz_nm * 1e-9
    grid = reference_grid(750e-9, max_optical_dz=max_dz)
    z = grid.z_edges
    i = int(np.searchsorted(z, 2.25e-6))
    j = i + 2
    np.testing.assert_allclose(np.diff(z[i - 2 : j + 3]), z[i + 1] - z[i], rtol=1e-12)
    planes = [float(z[i]), float(z[j])]
    monitors = [
        bz.FieldMonitor(
            center=(0, 0, p),
            size=(Lx, Ly, 0),
            freqs=frequencies,
            fields=("Ex",),
            name=f"E{k}",
        )
        for k, p in enumerate(planes)
    ]
    monitors.append(
        bz.FluxMonitor(
            center=(0, 0, Lz / 2 - d_absorber_space - 0.05e-6),
            size=(Lx, Ly, 0),
            freqs=frequencies,
            name="incident_plane",
        )
    )
    sim = make_reference_mesh_sim(
        lateral_spacing=750e-9,
        boundary=args.boundary,
        max_optical_dz=max_dz,
        reference=True,
        run_time=510e-15,
        monitors=monitors,
        sources=[
            bz.PlaneWaveSource(
                center=(0, 0, (Lz - d_absorber_space) / 2),
                size=(Lx, Ly, 0),
                direction="-z",
                source_time=bz.GaussianPulse(f0, f0 / 3, offset=5),
            )
        ],
    )
    result = sim.run(backend="jax", progress=False)
    e0 = result["E0"].get_dft_component("Ex").mean(axis=1)
    e1 = result["E1"].get_dft_component("Ex").mean(axis=1)
    h = z[i + 1] - z[i]
    forward, backward, k_fd = decompose_normal_air_wave(
        e0, e1, frequencies, dt=sim.dt, spacing=h, separation=planes[1] - planes[0]
    )
    impedance = np.sqrt(bz.MU_0 / bz.EPS_0)
    modal_power = 0.5 * Lx * Ly * abs(forward) ** 2 / impedance
    reflected_power = 0.5 * Lx * Ly * abs(backward) ** 2 / impedance
    plane_power = -np.asarray(result["incident_plane"].flux)
    previous = np.load(
        ROOT
        / "docs/reviews/cmos-polarized/reference-z-20nm-absorber/reference_spectra.npz"
    )
    error = float(np.max(abs(plane_power - previous["incident_plane"])))
    # A normally incident empty cell is independent of transverse grid density.
    # The historical reference used an accumulated float32 clock. Keep its
    # difference as evidence; it is not a same-clock regression assertion.
    out = (
        ROOT
        / "docs/reviews/cmos-isolation"
        / ("incident" if args.boundary == "absorber" else "incident-cpml")
    )
    if args.optical_dz_nm is not None:
        out = out.with_name(out.name + f"-dz{args.optical_dz_nm:g}nm")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "spectra.npz",
        wavelength_nm=bz.LIGHT_SPEED / frequencies * 1e9,
        modal_power=modal_power,
        backward_power=reflected_power,
        plane_power=plane_power,
        E0=e0,
        E1=e1,
        k_fd=k_fd,
        planes=planes,
    )
    report = dict(
        device=str(jax.devices()),
        clock_policy="integer_step",
        grid=list(sim.grid.shape),
        dt=sim.dt,
        steps=sim.num_steps,
        planes=planes,
        boundary=args.boundary,
        max_optical_dz_nm=args.optical_dz_nm,
        air_spacing_nm=float(h * 1e9),
        previous_reference_max_difference=error,
        modal_power_range=[float(modal_power.min()), float(modal_power.max())],
        backward_power_max=float(reflected_power.max()),
        interpolated_power_range=[float(plane_power.min()), float(plane_power.max())],
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
