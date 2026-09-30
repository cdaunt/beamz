"""Recompute sphere cross sections from saved complex surface fields."""

import argparse
import json
from pathlib import Path

import numpy as np
from metal_benchmarks import flux
from metal_far_field import sphere_projection


def analyze(path, correct_timing=False):
    path = Path(path)
    report = json.loads((path / "report.json").read_text())
    raw = dict(np.load(path / "surface_fields.npz"))
    spectra = dict(np.load(path / "spectra.npz"))
    if correct_timing:
        if report.get("magnetic_timing_aligned", False):
            raise ValueError("Magnetic timing already aligned")
        phase = np.exp(
            -1j * np.pi * 299792458 / (spectra["wavelength_nm"] * 1e-9) * report["dt"]
        )
        for key in raw:
            if key.rsplit("_", 1)[-1].startswith("H"):
                raw[key] = raw[key] * phase[:, None]
        np.savez_compressed(path / "surface_fields.npz", **raw)
    report["magnetic_timing_aligned"] = True
    incident = {
        c: raw["reference_incident_" + c] for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }
    intensity = 0.5 * (
        incident["Ex"] * incident["Hy"].conj() - incident["Ey"] * incident["Hx"].conj()
    ).real.mean(axis=1)
    spectra["intensity"] = intensity
    impedance_ratio = 376.730313668 * incident["Hy"] / incident["Ex"]
    report["incident_impedance_max_error"] = float(np.max(abs(impedance_ratio - 1)))
    report["incident_cross_polarization_max"] = float(
        np.max(
            np.sqrt(abs(incident["Ey"]) ** 2 + abs(incident["Ez"]) ** 2)
            / abs(incident["Ex"])
        )
    )
    report["surfaces"] = []
    for half in (6, 8):
        absorb = np.zeros(len(intensity))
        scatter = absorb.copy()
        empty = absorb.copy()
        for axis in range(3):
            for sign in (-1, 1):
                name = f"face-{half}-{axis}-{sign}"
                d = {
                    c: raw["device_" + name + "_" + c]
                    for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
                    if "device_" + name + "_" + c in raw
                }
                r = {c: raw["reference_" + name + "_" + c] for c in d}
                s = {c: d[c] - r[c] for c in d}
                w = raw["weights_" + name]
                absorb -= sign * (flux(d, axis, w) - flux(r, axis, w))
                scatter += sign * flux(s, axis, w)
                empty += sign * flux(r, axis, w)
        absorb /= intensity
        scatter /= intensity
        empty /= intensity
        spectra.update(
            {
                f"absorption_{half}": absorb,
                f"scattering_{half}": scatter,
                f"empty_flux_{half}": empty,
            }
        )
        report["surfaces"].append(
            dict(
                half_nm=half,
                absorption_relative_max=float(
                    max(abs(absorb - spectra["absorption_exact"]))
                    / max(spectra["absorption_exact"])
                ),
                scattering_relative_max=float(
                    max(abs(scatter - spectra["scattering_exact"]))
                    / max(spectra["scattering_exact"])
                ),
                absorption_peak_nm=float(spectra["wavelength_nm"][np.argmax(absorb)]),
                exact_peak_nm=float(
                    spectra["wavelength_nm"][np.argmax(spectra["absorption_exact"])]
                ),
                min_absorption=float(min(absorb)),
                min_scattering=float(min(scatter)),
            )
        )
    grid = np.load(path / "grid.npz")["x"]
    for row in report["surfaces"]:
        half = row["half_nm"]
        projected, extinction = sphere_projection(
            raw, grid, spectra["wavelength_nm"], intensity, half
        )
        spectra[f"scattering_projected_{half}"] = projected
        spectra[f"extinction_optical_theorem_{half}"] = extinction
        spectra[f"absorption_optical_theorem_{half}"] = extinction - projected
        row["absorption_optical_theorem_relative_max"] = float(
            max(abs(extinction - projected - spectra["absorption_exact"]))
            / max(spectra["absorption_exact"])
        )
        row["absorption_extraction_difference_relative_max"] = float(
            max(abs(extinction - projected - spectra[f"absorption_{half}"]))
            / max(spectra["absorption_exact"])
        )
        row["scattering_projected_relative_max"] = float(
            max(abs(projected - spectra["scattering_exact"]))
            / max(spectra["scattering_exact"])
        )
        if report.get("lateral_boundary") == "periodic":
            # Below diffraction, a square electric-dipole array radiates a
            # different power per cell than an isolated dipole of the same p.
            factor = (
                3
                * spectra["wavelength_nm"] ** 2
                / (4 * np.pi * report["actual_period_nm"] ** 2)
            )
            expected_array = factor * projected
            spectra[f"scattering_array_dipole_{half}"] = expected_array
            row["direct_vs_array_dipole_relative_max"] = float(
                max(abs(spectra[f"scattering_{half}"] - expected_array))
                / max(expected_array)
            )
            report["direct_scattering_observable"] = "periodic array power per cell"
    np.savez(path / "spectra.npz", **spectra)
    (path / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--correct-timing", action="store_true")
    a = p.parse_args()
    analyze(a.path, a.correct_timing)
