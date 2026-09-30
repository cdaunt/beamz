"""Reproducible reference checks; uses optional miepython for independent Mie QA."""

import json

import miepython
import numpy as np
from metal_benchmarks import DATA, material, mie
from PIL import Image

import beamz as bz


def main():
    checks = []
    imported = np.load(DATA / "tidy3d_permittivity.npz")
    for name in ("sphere_gold", "aluminum"):
        wl = np.linspace(400, 700, 100) * 1e-9
        eps = material(name).eps_model(bz.LIGHT_SPEED / wl)
        absorption, scattering = mie(eps, wl)
        ext, sca, *_ = miepython.efficiencies_mx(
            np.sqrt(eps).conj(), 2 * np.pi * 5e-9 / wl
        )
        row = dict(
            material=name,
            absorption_error=float(
                max(abs(absorption - (ext - sca) * np.pi * (5e-9) ** 2))
                / max(absorption)
            ),
            scattering_error=float(
                max(abs(scattering - sca * np.pi * (5e-9) ** 2)) / max(scattering)
            ),
        )
        if name in imported:
            row["permittivity_max_error"] = float(
                max(
                    abs(
                        material(name).eps_model(imported["frequency"]) - imported[name]
                    )
                )
            )
        checks.append(row)
    published = {}
    for kind, index in [("abs", 0), ("scat", 1)]:
        table = np.loadtxt(DATA / f"mie_plasmonic_flux_{kind}.txt", skiprows=1)
        # Tidy3D TFSF incident intensity is 1 W/um², so W equals cross section in um².
        exact = (
            mie(
                material("sphere_gold").eps_model(table[:, 0]),
                bz.LIGHT_SPEED / table[:, 0],
            )[index]
            * 1e12
        )
        published[kind] = dict(
            relative_max_error=float(max(abs(exact - table[:, 1])) / max(table[:, 1])),
            ratio_min=float(min(exact / table[:, 1])),
            ratio_max=float(max(exact / table[:, 1])),
        )
    original = complex(
        material("reflector_gold_snapshot").eps_model(bz.LIGHT_SPEED / 850e-9)
    )
    current = complex(material("reflector_gold").eps_model(bz.LIGHT_SPEED / 850e-9))
    report = dict(
        reflector_epsilon_850nm=dict(
            snapshot=[original.real, original.imag],
            sdk=[current.real, current.imag],
            snapshot_loss_reduction=1 - original.imag / current.imag,
        ),
        package="miepython",
        version=miepython.__version__,
        runs=checks,
        published_table_comparison=published,
    )
    (DATA / "mie_independent_check.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    assert all(
        r["absorption_error"] < 1e-8 and r["scattering_error"] < 1e-8 for r in checks
    )
    image = np.array(Image.open(DATA / "tidy3d_metasurface.png").convert("RGB")).astype(
        int
    )
    mask = (
        (image[:, :, 1] > image[:, :, 0] + 25)
        & (image[:, :, 1] > image[:, :, 2] + 20)
        & (image[:, :, 0] < 90)
    )
    rows = []
    for length in np.linspace(40, 280, 14):
        values = []
        for col, (x0, x1) in enumerate([(82, 402), (504, 824)]):
            x = round(x0 + (length - 40) / 240 * (x1 - x0))
            yy = np.where(mask[:, x])[0]
            yy = yy[(yy > 12) & (yy < 390)]
            y = float(np.median(yy))
            values.append(
                (377 - y) / 1.304 if col == 0 else 0.98 + (31 - y) * (0.02 / 65)
            )
        rows.append(
            dict(
                length_nm=float(length), phase_degrees=values[0], reflectivity=values[1]
            )
        )
    phase_origin = rows[0]["phase_degrees"]
    for row in rows:
        row["phase_degrees"] -= phase_origin
    (DATA / "metasurface_digitized.json").write_text(
        json.dumps(
            dict(
                source="tidy3d_metasurface.png",
                method="Green-line center pixels; axes tick calibration. Approximate image comparison, not solver arrays.",
                approximate_allowance=dict(phase_degrees=2, reflectivity=0.002),
                runs=rows,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
