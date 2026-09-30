"""Independent CPU Meep check of the small metal resonator.

A single Drude model matches the saved gold epsilon exactly at 850 nm. Exported
parameters are shared with BeamZ; off-center spectra test that common model,
not the original multi-pole fit. All interfaces are deliberately staircased.
"""

import argparse
import json
import time
from pathlib import Path

import meep as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/reviews/cmos-parity-followup/meep"
C = 299792458.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", type=float, default=5)
    parser.add_argument("--length", type=float, default=132.3076923076923)
    parser.add_argument("--coefficients-only", action="store_true")
    args = parser.parse_args()
    out = OUT / f"dx{args.dx:g}-length{args.length:g}"
    out.mkdir(parents=True, exist_ok=True)
    medium = json.loads(
        (
            ROOT / "examples/data/metal_benchmarks/tidy3d_reflector_snapshot.json"
        ).read_text()
    )["structures"][1]["medium"]
    f0 = 1 / 0.85
    omega = 2 * np.pi * C / 0.85e-6
    epsilon = complex(medium["eps_inf"])
    for a, c in medium["poles"]:
        a, c = complex(**a), complex(**c)
        epsilon += c / (-1j * omega - a) + c.conjugate() / (-1j * omega - a.conjugate())
    eps_inf = medium["eps_inf"]
    gamma = omega * epsilon.imag / (eps_inf - epsilon.real)
    plasma = np.sqrt((eps_inf - epsilon.real) * (omega**2 + gamma**2))
    gold = mp.Medium(
        epsilon=eps_inf,
        E_susceptibilities=[
            mp.DrudeSusceptibility(
                frequency=plasma / (2 * np.pi * C * 1e6),
                gamma=gamma / (2 * np.pi * C * 1e6),
                sigma=1,
            )
        ],
    )
    np.testing.assert_allclose(gold.epsilon(f0)[0, 0], epsilon, rtol=1e-12)
    structures = [
        mp.Block(
            center=mp.Vector3(0, 0, (-0.8 - 0.13) / 2),
            size=mp.Vector3(mp.inf, mp.inf, 0.8 - 0.13),
            material=mp.Medium(epsilon=2.25),
        ),
        mp.Block(
            center=mp.Vector3(0, 0, -0.065),
            size=mp.Vector3(mp.inf, mp.inf, 0.13),
            material=gold,
        ),
        mp.Block(
            center=mp.Vector3(0, 0, 0.025),
            size=mp.Vector3(mp.inf, mp.inf, 0.05),
            material=mp.Medium(epsilon=1.892),
        ),
    ]
    if args.length:
        structures.append(
            mp.Block(
                center=mp.Vector3(0, 0, 0.065),
                size=mp.Vector3(0.09, args.length * 0.001, 0.03),
                material=gold,
            )
        )
    source = mp.Source(
        mp.GaussianSource(f0, fwidth=f0 / 3, cutoff=5),
        component=mp.Ey,
        center=mp.Vector3(0, 0, 0.255),
        size=mp.Vector3(0.12, 0.3, 0),
    )

    def simulation(geometry):
        return mp.Simulation(
            cell_size=mp.Vector3(0.12, 0.3, 1.6),
            resolution=1000 / args.dx,
            boundary_layers=[mp.PML(0.2, direction=mp.Z)],
            sources=[source],
            geometry=geometry,
            eps_averaging=False,
            Courant=0.5,
            k_point=mp.Vector3(),
        )

    if args.coefficients_only:
        dev = simulation(structures)
        dev.init_sim()
        z = np.arange(-160, 111, args.dx / 2) * 0.001
        values = {}
        for c, axis in ((mp.Ex, mp.X), (mp.Ey, mp.Y), (mp.Ez, mp.Z)):
            values[str(c)] = np.array(
                [
                    1 / dev.fields.get_chi1inv(c, axis, mp.vec(0, 0, float(h)), f0)
                    for h in z
                ]
            )
        np.savez(out / "coefficients.npz", z_um=z, **values)
        beamz = (
            ROOT
            / f"docs/reviews/cmos-parity-followup/beamz/dx{args.dx:g}-length{args.length:g}/coefficients.npz"
        )
        if beamz.exists():
            saved = np.load(beamz)
            compare = {}
            for name, c, axis in (
                ("Ex", mp.Ex, mp.X),
                ("Ey", mp.Ey, mp.Y),
                ("Ez", mp.Ez, mp.Z),
            ):
                coords = [saved[name + "_" + a] * 1e6 for a in "zyx"]
                samples = np.array(
                    [
                        1
                        / dev.fields.get_chi1inv(
                            c, axis, mp.vec(float(x), float(y), float(z)), f0
                        )
                        for z in coords[0]
                        for y in coords[1]
                        for x in coords[2]
                    ]
                ).reshape(saved[name].shape)
                compare[name] = samples
                difference = abs(samples - saved[name]) > 0.01
                print(
                    name,
                    "different",
                    np.count_nonzero(difference),
                    "of",
                    samples.size,
                    flush=True,
                )
            np.savez(out / "beamz_sites.npz", **compare)

        return

    ref = simulation([])
    r = ref.add_flux(
        f0,
        0.2 * f0,
        41,
        mp.FluxRegion(center=mp.Vector3(0, 0, 0.51), size=mp.Vector3(0.12, 0.3, 0)),
    )
    inc = ref.add_flux(
        f0,
        0.2 * f0,
        41,
        mp.FluxRegion(center=mp.Vector3(0, 0, 0.18), size=mp.Vector3(0.12, 0.3, 0)),
    )
    start = time.monotonic()
    ref.run(until=300e-15 * C * 1e6)
    incident = -np.array(mp.get_fluxes(inc))
    subtract = ref.get_flux_data(r)
    freqs = np.array(mp.get_flux_freqs(r))
    ref.reset_meep()
    dev = simulation(structures)
    r = dev.add_flux(
        f0,
        0.2 * f0,
        41,
        mp.FluxRegion(center=mp.Vector3(0, 0, 0.51), size=mp.Vector3(0.12, 0.3, 0)),
    )
    t = dev.add_flux(
        f0,
        0.2 * f0,
        41,
        mp.FluxRegion(center=mp.Vector3(0, 0, -0.3), size=mp.Vector3(0.12, 0.3, 0)),
    )
    dev.load_minus_flux_data(r, subtract)
    dev.run(until=225e-15 * C * 1e6)
    early = np.array(mp.get_fluxes(r))
    dev.run(until=75e-15 * C * 1e6)
    reflection = np.array(mp.get_fluxes(r)) / incident
    transmission = -np.array(mp.get_fluxes(t)) / incident
    report = dict(
        meep_version=mp.__version__,
        dx_nm=args.dx,
        length_nm=args.length,
        metal_model="Drude matched at 850 nm",
        epsilon_850=[epsilon.real, epsilon.imag],
        epsilon_inf=eps_inf,
        gamma_rad_s=float(gamma),
        plasma_rad_s=float(plasma),
        eps_averaging=False,
        runtime_fs=300,
        elapsed_s=time.monotonic() - start,
        reflection_850=float(reflection[20]),
        transmission_850=float(transmission[20]),
        settling_max=float(np.max(abs(np.array(mp.get_fluxes(r)) - early) / incident)),
    )
    np.savez(
        out / "spectra.npz",
        frequency_hz=freqs * C * 1e6,
        reflection=reflection,
        transmission=transmission,
        incident=incident,
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
