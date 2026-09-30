"""Summarize and verify the saved junction, loss, and registration controls."""

import csv
import json

import numpy as np
from metal_benchmarks import OUT


def write_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = []
    for path in sorted((OUT / "junction-loss").glob("*/*/report.json")):
        report = json.loads(path.read_text())
        run = report["runs"][0]
        e = run["execution"]
        drift = e.get(
            "native_electric_checkpoint_drift", e.get("metal_dft_checkpoint_drift")
        )
        row = dict(
            variant=path.parents[1].name,
            mesh=report["mesh"],
            dx_nm=report["dx_nm"] if report["mesh"] == "refined" else "graded",
            length_nm=run["length_nm"],
            reflectivity_percent=100 * e["reflection"],
            material_absorption_percent=100 * e["absorption"],
            native_transmission_percent=100 * e["native_transmission"],
            native_balance_residual_pp=100 * e["native_energy_residual"],
            global_balance_residual_pp=100 * e["energy_residual"],
            junction_candidate_sites=sum(
                v["junction_sites"] for v in e["junction"].values()
            ),
            changed_dielectric_sites=(
                0
                if path.parents[1].name == "baseline"
                else sum(v["changed_sites"] for v in e["junction"].values())
            ),
            max_native_electric_drift=max(drift.values()),
        )
        assert "cuda" in report["device"].lower()
        assert np.isfinite(list(e["absorption_components"].values())).all()
        assert min(e["absorption_components"].values()) >= 0
        assert abs(e["native_energy_residual"]) < 1e-5, row
        assert max(drift.values()) < 1e-4, row
        if row["mesh"] == "refined":
            assert abs(e["energy_residual"]) < 1e-4, row
        rows.append(row)
    assert len(rows) == 5, (
        "Expected planar control and two junction policies on two meshes"
    )
    write_csv(OUT / "junction_loss_summary.csv", rows)
    registrations = []
    baseline = next(
        r for r in rows if r["mesh"] == "refined" and r["variant"] == "baseline"
    )
    for path in sorted((OUT / "registration").glob("*/*/report.json")):
        report = json.loads(path.read_text())
        run = report["runs"][0]
        e = run["execution"]
        unshifted = baseline["reflectivity_percent"]
        if report["dx_nm"] != 5:
            old = json.loads(
                (
                    OUT
                    / f"reflector-snapshot-{report['dx_nm']:g}nm-300fs-outgoing/report.json"
                ).read_text()
            )
            unshifted = 100 * next(
                row["r_power"] for row in old["runs"] if row["length_nm"] > 0
            )
        registrations.append(
            dict(
                dx_nm=report["dx_nm"],
                xy_shift_nm=e["xy_shift_nm"],
                z_shift_nm=e["z_shift_nm"],
                reflectivity_percent=100 * run["r_power"],
                change_from_unshifted_pp=100 * run["r_power"] - unshifted,
                max_native_electric_drift=e["checkpoint_to_full_dft_relative_change"][
                    "E"
                ],
            )
        )
        assert registrations[-1]["max_native_electric_drift"] < 1e-4
    assert len(registrations) == 4, "Expected three 5 nm controls and one 2.5 nm repeat"
    write_csv(OUT / "registration_summary.csv", registrations)
    print(json.dumps(dict(junction_loss=rows, registration=registrations), indent=2))


if __name__ == "__main__":
    main()
