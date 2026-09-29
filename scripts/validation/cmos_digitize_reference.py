"""Recover approximate spectra from the published CMOS plot, without cloud data."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/reviews/cmos-reverse-engineering"


def refractive_index(spec, frequencies):
    s = -2j * np.pi * frequencies
    epsilon = np.full(s.shape, spec["epsilon_inf"], dtype=complex)
    for pole, residue in spec["poles"]:
        a, c = complex(*pole), complex(*residue)
        epsilon += c / (s - a) + c.conjugate() / (s - a.conjugate())
    return np.sqrt(epsilon)


def compare_filter_fits():
    """Check snapshot n/k against the independently published fitted curves."""
    out = OUT
    pix = np.asarray(
        Image.open(out / "published_filter_fits.png").convert("RGB")
    ).astype(float)
    materials = json.loads((ROOT / "examples/data/cmos_rgb/filters.json").read_text())
    report = {}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    raw = {}
    for ax, (c, x0, x1, yt, topvalue) in zip(
        axes,
        [
            ("red", 146, 646, 79, 1.75),
            ("green", 826, 1325, 103, 1.75),
            ("blue", 1506, 2005, 80, 1.6),
        ],
        strict=True,
    ):
        scale = pix.shape[1] / 2048
        x0, x1, yt, ybottom = [round(v * scale) for v in (x0, x1, yt, 555)]
        x = np.arange(x0 + 3, x1 - 2)
        wl = 400 + (x - x0) * 300 / (x1 - x0)
        region = pix[: ybottom + 1, x]
        masks = {
            "n": (region[..., 1] > 65) & (region[..., 0] < 80) & (region[..., 2] < 80),
            "k": (region[..., 0] > 220)
            & (region[..., 1] > 100)
            & (region[..., 1] < 205)
            & (region[..., 2] < 70),
        }
        nk = refractive_index(materials[c], 299792458 / (wl * 1e-9))
        for name, mask in masks.items():
            # Exclude legend samples: n curve stays above 1, k below .6.
            value = (ybottom - np.arange(ybottom + 1)) * topvalue / (ybottom - yt)
            mask &= ((value > 1) if name == "n" else (value < 0.6))[:, None]
            samples = np.full(len(x), np.nan)
            for i in range(len(x)):
                ys = np.flatnonzero(mask[:, i])
                if len(ys) and np.ptp(ys) < 18:
                    samples[i] = (ybottom - np.median(ys)) * topvalue / (ybottom - yt)
            expected = nk.real if name == "n" else nk.imag
            valid = np.isfinite(samples)
            delta = np.abs(samples[valid] - expected[valid])
            report[c + "_" + name] = {
                "median_abs_error": float(np.median(delta)),
                "p95_abs_error": float(np.quantile(delta, 0.95)),
            }
            ax.plot(
                wl,
                samples,
                color="green" if name == "n" else "orange",
                label=f"Published {name}",
            )
            ax.plot(wl, expected, "--", color="black", lw=1, label=f"Snapshot {name}")
            raw[c + "_" + name] = samples
        raw[c + "_wavelength_nm"] = wl
        ax.set(title=c, xlabel="Wavelength (nm)", ylabel="n, k")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.savefig(out / "filter_fit_crosscheck.png", dpi=150)
    np.savez(out / "digitized_filter_fits.npz", **raw)
    (out / "filter_fit_crosscheck.json").write_text(json.dumps(report, indent=2) + "\n")


def compare_field_display():
    """Apply the reference plotting convention to already saved BeamZ fields."""
    p = OUT
    d = np.load(ROOT / "docs/reviews/cmos-isolation/notebook/field_xy_silicon.npz")
    nx, ny = len(d["x_edges_um"]) - 1, len(d["y_edges_um"]) - 1
    normalized = (
        sum(abs(d[c]) ** 2 for c in ("Ex", "Ey", "Ez")).reshape(3, ny, nx)
        / d["incident_intensity"][:, None, None]
    )
    # One watt over 3x3 um²; electric fields in sqrt(W*ohm)/um.
    intensity = normalized * (2 * 376.730313668 / 9)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    stats = {}
    for ax, c, values in zip(axes, ("red", "green", "blue"), intensity, strict=True):
        vmin, vmax = np.quantile(values, [0.02, 0.98])
        im = ax.pcolormesh(
            d["x_edges_um"], d["y_edges_um"], values, cmap="magma", vmin=vmin, vmax=vmax
        )
        ax.set(title=c, xlabel="x (µm)", ylabel="y (µm)", aspect="equal")
        fig.colorbar(im, ax=ax, extend="both", label="|E|² (1 W/cell, µm units)")
        stats[c] = {
            "percentile_2": float(vmin),
            "percentile_98": float(vmax),
            "maximum": float(values.max()),
        }
    fig.suptitle("BeamZ fields with Tidy3D’s per-panel robust display convention")
    fig.savefig(p / "matched_field_display.png", dpi=150)
    (p / "field_display_report.json").write_text(
        json.dumps(
            {
                "normalization": "1 W per 3x3 um² cell; |E|² = normalized intensity * 2 Z0 / 9",
                "display": "per-panel 2nd and 98th percentiles; published colorbars are clipped, not maxima",
                "statistics": stats,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    pixels = np.asarray(Image.open(OUT / "published_efficiency.png").convert("RGB"))
    # Original 928x723 figure; axis/tick locations calibrated from its pixels.
    left, right, top, bottom = 135, 878, 29, 620
    assert pixels.shape == (723, 928, 3)
    x = np.arange(left + 3, right - 2)
    wavelengths = 400 + (x - left) * 300 / (right - left)
    y = np.arange(top + 3, bottom - 2)
    rgb = pixels[y[:, None], x[None, :]].astype(float)
    masks = {
        "red": (rgb[..., 0] > 180) & (rgb[..., 1] < 100) & (rgb[..., 2] < 100),
        "green": (rgb[..., 1] > 70) & (rgb[..., 0] < 80) & (rgb[..., 2] < 80),
        "blue": (rgb[..., 2] > 180) & (rgb[..., 0] < 100) & (rgb[..., 1] < 100),
    }
    guides = dict(red=755, green=507, blue=259)
    extracted = {"wavelength_nm": wavelengths}
    report = {
        "source": "https://www.flexcompute.com/tidy3d/examples/notebooks/CMOSRGBSensor/",
        "method": "RGB segmentation; median curve row per column; dashed guides excluded; clipped/axis-overlapped samples omitted",
        "axis_pixels": dict(left=left, right=right, top=top, bottom=bottom),
        "reading_uncertainty_pp": 0.15,
        "channels": {},
    }
    baseline = np.load(
        ROOT / "docs/reviews/cmos-isolation/components/full/calibrated_efficiency.npz"
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(12, 3.7), sharey=True, constrained_layout=True
    )
    for ax, (channel, mask) in zip(axes, masks.items(), strict=True):
        values = np.full(x.size, np.nan)
        for i, column in enumerate(mask.T):
            rows = y[column]
            if abs(x[i] - guides[channel]) <= 4 or not len(rows):
                continue
            # Reject vertical guide remnants and clipping at the upper axis.
            if np.ptp(rows) > 12 or rows.min() <= top + 3:
                continue
            values[i] = (bottom - np.median(rows)) * 0.25 / (bottom - top)
        extracted[channel] = values
        valid = np.isfinite(values)
        center = dict(red=650, green=550, blue=450)[channel]
        target = float(np.interp(center, wavelengths[valid], values[valid]))
        sim_wl = baseline["wavelength_nm"]
        order = np.argsort(sim_wl)
        simulation = np.interp(wavelengths, sim_wl[order], baseline[channel][order])
        band = valid & (values > 0.025)
        ratios = simulation[band] / values[band]
        report["channels"][channel] = dict(
            central_percent=100 * target,
            passband_median_ratio=float(np.median(ratios)),
            passband_ratio_p10_p90=np.quantile(ratios, [0.1, 0.9]).tolist(),
            passband_rms_difference_pp=float(
                100 * np.sqrt(np.mean((simulation[band] - values[band]) ** 2))
            ),
        )
        ax.plot(
            wavelengths, 100 * values, color=channel, label="Digitized published plot"
        )
        ax.plot(
            sim_wl[order],
            100 * baseline[channel][order],
            "--",
            color=channel,
            label="Corrected BeamZ",
        )
        ax.fill_between(
            wavelengths,
            100 * values - 0.15,
            100 * values + 0.15,
            color=channel,
            alpha=0.2,
        )
        ax.set(title=channel, xlabel="Wavelength (nm)", ylim=(0, 28))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Optical efficiency (%)")
    axes[1].legend(fontsize=9)
    fig.savefig(OUT / "digitized_comparison.png", dpi=150)
    np.savez(OUT / "digitized_efficiency.npz", **extracted)
    (OUT / "digitization_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
    compare_filter_fits()
    compare_field_display()
