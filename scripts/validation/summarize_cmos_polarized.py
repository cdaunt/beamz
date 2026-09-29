"""Calibrate completed broadband runs without assuming wavelength-array order."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/reviews/cmos-polarized"
CHANNELS = (
    ("red", 650, "#bb3535"),
    ("green", 550, "#267a36"),
    ("blue", 450, "#315cb3"),
)


def main():
    runs = {}
    curves = {}
    for path in sorted(OUTPUT.glob("*/sensor_spectra.npz")):
        folder = path.parent
        if not (folder / "reference_spectra.npz").exists():
            continue
        with np.load(path) as device, np.load(folder / "reference_spectra.npz") as ref:
            np.testing.assert_allclose(device["wavelength_nm"], ref["wavelength_nm"])
            incident = ref["incident_plane"]
            if not np.isfinite(incident).all() or np.any(incident <= 0):
                raise ValueError(f"Invalid reference spectrum: {folder}")
            wl = device["wavelength_nm"]
            channels = {
                c: (device["green1"] + device["green2"] if c == "green" else device[c])
                / incident
                for c, _, _ in CHANNELS
            }
            np.savez(folder / "calibrated_efficiency.npz", wavelength_nm=wl, **channels)
        order = np.argsort(wl)
        report = json.loads((folder / "sensor_report.json").read_text())
        report["central_efficiency_percent"] = {
            c: float(100 * np.interp(w, wl[order], channels[c][order]))
            for c, w, _ in CHANNELS
        }
        changes = report["temporal_max_absolute_change"]
        report["temporal_change_upper_bound_pp"] = {
            c: float(
                100
                * (
                    changes["green1"] + changes["green2"]
                    if c == "green"
                    else changes[c]
                )
                / incident.min()
            )
            for c, _, _ in CHANNELS
        }
        runs[folder.name] = report
        curves[folder.name] = dict(wavelength_nm=wl, **channels)
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        for c, _, color in CHANNELS:
            ax.plot(wl[order], 100 * channels[c][order], color=color, label=c)
        ax.set(
            xlabel="Wavelength (nm)",
            ylabel="Optical efficiency (%)",
            title=folder.name,
            xlim=(400, 700),
            ylim=(0, None),
        )
        ax.legend()
        ax.grid(alpha=0.2)
        fig.savefig(folder / "calibrated_efficiency.png", dpi=160)
        plt.close(fig)
    old = np.load(
        ROOT
        / "docs/reviews/cmos-reference-parity/notebook25nm/calibrated_efficiency.npz"
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, (c, w, _) in zip(axes, CHANNELS, strict=True):
        order = np.argsort(old["wavelength_nm"])
        ax.plot(
            old["wavelength_nm"][order],
            100 * old[c][order],
            color="#777777",
            linestyle="--",
            label="Historical volume, 25 nm",
        )
        for i, (name, data) in enumerate(curves.items()):
            order = np.argsort(data["wavelength_nm"])
            ax.plot(
                data["wavelength_nm"][order],
                100 * data[c][order],
                label=name,
                color=("#315cb3", "#ab6a24", "#647c46", "#aa4670")[i % 4],
            )
        ax.scatter(
            [w],
            [20 if c == "green" else 6],
            color="black",
            marker="x",
            zorder=5,
            label="Tidy3D plot estimate",
        )
        ax.set(
            title=c,
            xlabel="Wavelength (nm)",
            ylabel="Optical efficiency (%)",
            xlim=(400, 700),
            ylim=(0, None),
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.savefig(OUTPUT / "comparison.png", dpi=160)
    plt.close(fig)
    summary = dict(
        runs=runs,
        reference_output="Visual plot estimates only; original numerical output spectra are unavailable.",
    )
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: v["central_efficiency_percent"] for k, v in runs.items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
