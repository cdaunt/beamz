"""Audit diagonal truncations against Deinega & Valuev (2007), Eq. (1).

This is a constitutive algebra comparison, not a scattering simulation.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cmos_digitize_reference import refractive_index

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/reviews/cmos-reverse-engineering"


def main():
    materials = json.loads((ROOT / "examples/data/cmos_rgb/materials.json").read_text())
    frequencies = np.array([299792458 / 550e-9])
    silica = refractive_index(materials["SiO2_Palik_LowLoss"], frequencies)[0] ** 2
    metal = refractive_index(materials["Al_Rakic1995"], frequencies)[0] ** 2
    report = {"wavelength_nm": 550, "material_fraction": 0.5, "cases": {}}
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for name, e1, e2 in (
        ("air_silica", 1, silica),
        ("silica_aluminum", silica, metal),
        ("static_1_12", 1, 12),
    ):
        arithmetic = (e1 + e2) / 2
        harmonic = 2 / (1 / e1 + 1 / e2)
        angles = np.linspace(0, 90, 181)
        n2 = np.cos(np.deg2rad(angles)) ** 2
        current_beta = 1 / ((1 - n2) * arithmetic + n2 * harmonic)
        paper_beta_diagonal = (1 - n2) / arithmetic + n2 / harmonic
        np.testing.assert_allclose(current_beta[[0, -1]], paper_beta_diagonal[[0, -1]])
        normal = np.array([1, 1, 0]) / np.sqrt(2)
        projection = np.outer(normal, normal)
        epsilon = harmonic * projection + arithmetic * (np.eye(3) - projection)
        inverse = projection / harmonic + (np.eye(3) - projection) / arithmetic
        np.testing.assert_allclose(np.linalg.inv(epsilon), inverse, atol=1e-12)

        def pair(value):
            return [float(np.real(value)), float(np.imag(value))]

        report["cases"][name] = {
            "constituent_epsilon": [pair(e1), pair(e2)],
            "current_effective_epsilon_x_at_45deg": pair(1 / current_beta[90]),
            "inverse_diagonal_effective_epsilon_x_at_45deg": pair(
                1 / paper_beta_diagonal[90]
            ),
            "full_tensor_inverse_xy_at_45deg": pair(inverse[0, 1]),
            "relative_update_coefficient_difference_at_45deg": float(
                abs(current_beta[90] / paper_beta_diagonal[90] - 1)
            ),
        }
        ax.plot(angles, 100 * abs(current_beta / paper_beta_diagonal - 1), label=name)
    ax.set(
        xlabel="Angle between interface normal and Ex (degrees)",
        ylabel="Relative difference in inverse coefficient (%)",
        title="Diagonalizing ε and ε⁻¹ are different approximations",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(OUT / "interface_law_comparison.png", dpi=150)
    (OUT / "interface_law_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
