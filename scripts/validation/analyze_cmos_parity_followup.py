"""Recompute comparisons from the saved CMOS and independent Meep controls."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/reviews/cmos-parity-followup"


def main():
    def report(path):
        return json.loads(path.read_text())

    baseline = report(
        ROOT
        / "docs/reviews/cmos-isolation/components/full-periodic-supports/report.json"
    )["central_efficiency_percent"]
    volume = report(OUT / "full-volume/report.json")["central_efficiency_percent"]
    frozen = report(OUT / "frozen-450nm-dx20nm/report.json")
    closed = report(OUT / "frozen-450nm-dx20nm-closed-metal-faces/report.json")
    meep_path = OUT / "meep/dx10-length132.308"
    beamz_path = OUT / "beamz/dx10-length132.308-meep-ownership"
    m, b = report(meep_path / "report.json"), report(beamz_path / "report.json")
    for key in (
        "dx_nm",
        "length_nm",
        "epsilon_850",
        "epsilon_inf",
        "gamma_rad_s",
        "plasma_rad_s",
    ):
        np.testing.assert_allclose(m[key], b[key], rtol=1e-12)
    ms, bs = np.load(meep_path / "spectra.npz"), np.load(beamz_path / "spectra.npz")
    np.testing.assert_allclose(ms["frequency_hz"], bs["frequency_hz"], rtol=1e-12)
    delta = 100 * (bs["reflection"] - ms["reflection"])
    native = report(OUT / "beamz/dx10-length132.308/report.json")
    native_coefficients = np.load(OUT / "beamz/dx10-length132.308/coefficients.npz")
    meep_coefficients = np.load(meep_path / "beamz_sites.npz")
    ownership = {}
    for component in ("Ex", "Ey", "Ez"):
        different = (
            abs(native_coefficients[component] - meep_coefficients[component]) > 0.01
        )
        ownership[component] = dict(
            changed_sites=int(different.sum()),
            z_nm=(
                native_coefficients[component + "_z"][np.any(different, axis=(1, 2))]
                * 1e9
            ).tolist(),
        )
    summary = dict(
        volume_minus_diagonal_pp={c: volume[c] - baseline[c] for c in baseline},
        frozen_minus_broadband_blue_pp=frozen["efficiency_percent"]["blue"]
        - baseline["blue"],
        closed_faces_minus_native_blue_pp=closed["efficiency_percent"]["blue"]
        - frozen["efficiency_percent"]["blue"],
        meep=dict(
            unmatched_central_difference_pp=100
            * (native["reflection_850"] - m["reflection_850"]),
            matched_central_difference_pp=float(delta[20]),
            max_abs_spectral_difference_pp=float(abs(delta).max()),
            rms_spectral_difference_pp=float(np.sqrt(np.mean(delta**2))),
            frequency_count=len(delta),
            ownership_audit=ownership,
        ),
        exact_mesh_completed=False,
    )
    fine_path = OUT / "frozen-450nm-dx10nm-chunked"
    checkpoint = report(fine_path / "checkpoint.json")
    fine = dict(
        grid=[300, 300, 430],
        checkpoint_blue_percent=100
        * checkpoint["fluxes"]["blue"]
        / frozen["incident_power"],
        checkpoint_time_fs=checkpoint["time_fs"],
        completed_510fs=False,
    )
    if (fine_path / "report.json").exists():
        final = report(fine_path / "report.json")
        fine.update(
            completed_510fs=True,
            blue_percent=final["efficiency_percent"]["blue"],
            fine_minus_coarse_blue_pp=final["efficiency_percent"]["blue"]
            - frozen["efficiency_percent"]["blue"],
            checkpoint_to_final_blue_change_pp=final["efficiency_percent"]["blue"]
            - fine["checkpoint_blue_percent"],
        )
    summary["lateral_10nm"] = fine
    exact = OUT / "frozen-450nm-dx20nm-reference-xy/report.json"
    if exact.exists():
        r = report(exact)
        summary["exact_mesh_completed"] = True
        summary["exact_mesh_blue_percent"] = r["efficiency_percent"]["blue"]
        summary["exact_mesh_minus_coarse_blue_pp"] = (
            r["efficiency_percent"]["blue"] - frozen["efficiency_percent"]["blue"]
        )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
