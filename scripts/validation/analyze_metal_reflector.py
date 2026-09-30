"""Planar Fresnel and two-plane checks for the metal antenna control."""

import json

import numpy as np
from metal_benchmarks import OUT, material

import beamz as bz


def planar_reflection(gold_model="sdk"):
    wavelength = 850e-9
    matrix = np.eye(2, dtype=complex)
    for epsilon, depth in [
        (1.892, 50e-9),
        (
            material(
                "reflector_gold_snapshot"
                if gold_model == "snapshot"
                else "reflector_gold"
            ).eps_model(bz.LIGHT_SPEED / wavelength),
            130e-9,
        ),
    ]:
        n = np.sqrt(epsilon + 0j)
        phase = 2 * np.pi * n * depth / wavelength
        layer = np.array(
            [
                [np.cos(phase), -1j * np.sin(phase) / n],
                [-1j * n * np.sin(phase), np.cos(phase)],
            ]
        )
        matrix = matrix @ layer
    b = matrix[0, 0] + 1.5 * matrix[0, 1]
    c = matrix[1, 0] + 1.5 * matrix[1, 1]
    return (b - c) / (b + c), 1.5 * abs(2 / (b + c)) ** 2


def main():
    rows = []
    paths = list(OUT.glob("reflector-*/report.json")) + list(
        OUT.glob("diagnostics/prior_reflection_measurements/reflector-*/report.json")
    )
    for path in sorted(paths):
        report = json.loads(path.read_text())
        r_exact, t_exact = planar_reflection(report.get("gold_model", "sdk"))
        outgoing = report.get("reflection_measurement") == "above_source_native_power"
        planes = report.get("monitor_z_nm", {"r": 150, "r2": 200})
        for row in report["runs"]:
            amp = complex(row["r_real"], row["r_imag"])
            amp2 = complex(row["r2_real"], row["r2_imag"])
            result = dict(
                directory=str(path.parent.relative_to(OUT)),
                dx_nm=report["dx_nm"],
                mesh=report.get("mesh", "refined"),
                measurement=report.get("reflection_measurement", "subtracted_flux"),
                gold_model=report.get("gold_model", "sdk"),
                length_nm=row["length_nm"],
                reflectivity=row["r_power"],
                transmission=row["transmission"],
                absorption=1 - row["r_power"] - row["transmission"],
                R_planes_difference_pp=100 * (row["r_power"] - row["r2_power"]),
                phase_propagation_error_degrees=float(
                    np.angle(
                        amp2
                        / amp
                        * np.exp(
                            -(1j if outgoing else 2j)
                            * 2
                            * np.pi
                            * (planes["r2"] - planes["r"])
                            / 850
                        )
                    )
                    * 180
                    / np.pi
                ),
            )
            if not outgoing:
                result["power_vs_zero_order_amplitude_pp"] = 100 * (
                    row["r_power"] - abs(amp) ** 2
                )
            if row["length_nm"] == 0:
                path_nm = (
                    planes["r"] + planes["input0"] - 100
                    if outgoing
                    else 2 * (planes["r"] - 50)
                )
                expected = r_exact * np.exp(1j * 2 * np.pi * path_nm / 850)
                result.update(
                    R_exact=float(abs(r_exact) ** 2),
                    T_exact=float(t_exact),
                    R_error_pp=float(100 * (row["r_power"] - abs(r_exact) ** 2)),
                    phase_error_degrees=float(np.angle(amp / expected) * 180 / np.pi),
                )
            rows.append(result)
    (OUT / "reflector_checks.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
