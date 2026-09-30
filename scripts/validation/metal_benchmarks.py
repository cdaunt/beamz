"""Local GPU metal controls: sphere/Mie and periodic antenna reflection.

Spheres use matched empty-domain fields; antennas use above-source reflection.
All distances in SI internally; source material snapshots are checked in.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np
from scipy.special import spherical_jn, spherical_yn

import beamz as bz
from beamz.design.raster import Box, Grid, Object, RasterOptions, Scene, Sphere

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "examples/data/metal_benchmarks"
OUT = ROOT / "docs/reviews/metal-benchmarks"
sys.path.insert(0, str(ROOT / "examples"))
from cmos_rgb_sensor import load_materials  # noqa: E402


def material(name):
    if name == "aluminum":
        return load_materials()["metal"]
    if name == "reflector_gold_snapshot":
        raw = json.loads((DATA / "tidy3d_reflector_snapshot.json").read_text())[
            "structures"
        ][1]["medium"]
    else:
        raw = json.loads((DATA / "tidy3d_gold.json").read_text())["materials"][name]
    return bz.PoleResidue(
        raw["eps_inf"],
        [(complex(**a), complex(**c)) for a, c in raw["poles"]],
        frequency_range=raw["frequency_range"],
    )


def mie(epsilon, wavelengths, radius=5e-9):
    """Full Mie cross sections in m², outgoing Hankel, exp(-i omega t)."""
    x = 2 * np.pi * radius / np.asarray(wavelengths)
    m = np.sqrt(np.asarray(epsilon, dtype=complex))
    scattering = np.zeros_like(x)
    extinction = np.zeros_like(x)
    for n in range(1, int(np.ceil(max(x + 4 * x ** (1 / 3) + 2))) + 1):
        mx = m * x
        p = x * spherical_jn(n, x)
        dp = spherical_jn(n, x) + x * spherical_jn(n, x, derivative=True)
        q = mx * spherical_jn(n, mx)
        dq = spherical_jn(n, mx) + mx * spherical_jn(n, mx, derivative=True)
        h = x * (spherical_jn(n, x) + 1j * spherical_yn(n, x))
        dh = dp + 1j * (spherical_yn(n, x) + x * spherical_yn(n, x, derivative=True))
        a = (m * q * dp - p * dq) / (m * q * dh - h * dq)
        b = (q * dp - m * p * dq) / (q * dh - m * h * dq)
        scattering += (2 * n + 1) * (abs(a) ** 2 + abs(b) ** 2)
        extinction += (2 * n + 1) * (a + b).real
    scale = np.asarray(wavelengths) ** 2 / (2 * np.pi)
    return scale * (extinction - scattering), scale * scattering


def graded_axis(dx_nm, half_nm=450, fine_nm=9, maximum_nm=12):
    n = int(np.ceil(fine_nm / dx_nm))
    positive = list(np.arange(n + 1) * dx_nm)
    step = dx_nm
    while positive[-1] < half_nm:
        step = min(step * 1.2, maximum_nm)
        positive.append(positive[-1] + step)
    # Last cell is merged if tiny; outer boundary lies beyond requested padding.
    values = np.asarray(positive) * 1e-9
    return np.concatenate((-values[:0:-1], values))


def fields(result):
    values = {c: np.asarray(result.get_dft_component(c)) for c in result.dft_fields}
    # Public field DFTs retain Yee's half-step H timing. Match the flux API.
    phase = np.exp(-1j * np.pi * result.get_dft_frequencies() * result.dft_base_dt)
    for c in values:
        if c.startswith("H"):
            values[c] = values[c] * phase[:, None]
    return values


def flux(field, axis, weights):
    a, b = (("y", "z"), ("z", "x"), ("x", "y"))[axis]
    p = field["E" + a] * field["H" + b].conj() - field["E" + b] * field["H" + a].conj()
    return 0.5 * np.sum(p.real * weights, axis=1)


def execute(
    scene,
    grid,
    sources,
    monitors,
    boundaries,
    runtime_fs,
    courant=0.8,
    smoothing="farjadpour_diagonal",
):
    dt = grid.cfl_time_step(courant)
    sim = bz.Simulation(
        scene=scene,
        raster_grid=grid,
        raster_options=RasterOptions(smoothing=smoothing, metal_smoothing="staircase"),
        sources=sources,
        monitors=monitors,
        boundaries=boundaries,
        time=np.arange(0, runtime_fs * 1e-15 + dt / 2, dt),
    )
    print(
        json.dumps(dict(grid=list(grid.shape), dt=dt, steps=sim.num_steps)), flush=True
    )
    start = time.monotonic()
    first = sim.advance(
        num_steps=3 * sim.num_steps // 4,
        backend="jax",
        progress=False,
        donate_state=True,
    )
    result = sim.advance(
        state=first.state, backend="jax", progress=False, donate_state=True
    ).results
    drift = {}
    old_results = first.results.renormalize(None)
    new_results = result.renormalize(None)
    for family in ("E", "H"):
        numerator = denominator = 0.0
        for monitor in monitors:
            old, new = old_results[monitor.name], new_results[monitor.name]
            for c in new.dft_fields:
                if c.startswith(family):
                    a = old.get_dft_component(c) * old.dft_weight_sum[:, None]
                    b = new.get_dft_component(c) * new.dft_weight_sum[:, None]
                    numerator = max(numerator, float(np.max(abs(a - b))))
                    denominator = max(denominator, float(np.max(abs(b))))
        drift[family] = numerator / denominator if denominator else 0.0
    for monitor in monitors:
        if isinstance(monitor, bz.FluxMonitor):
            measured = result[monitor.name]
            np.testing.assert_allclose(
                flux(
                    fields(measured), measured.normal_axis, measured.integration_weights
                ),
                measured.flux / measured.normal_sign,
                rtol=1e-10,
                atol=1e-30,
            )
    print(f"Finished in {time.monotonic() - start:.1f}s", flush=True)
    return result, dict(
        grid=list(grid.shape),
        dt=dt,
        steps=sim.num_steps,
        elapsed_s=time.monotonic() - start,
        checkpoint_fraction=0.75,
        checkpoint_to_full_dft_relative_change=drift,
    )


def sphere(args):
    out = (
        OUT
        / f"sphere-{args.material}-{args.dx:g}nm-{args.runtime:g}fs-{args.lateral}-period{args.period:g}nm-alpha{args.cpml_alpha:g}"
    )
    out.mkdir(parents=True, exist_ok=True)
    f0 = bz.LIGHT_SPEED / 530e-9
    freq = np.linspace(0.8 * f0, 1.2 * f0, 31)
    if args.material == "aluminum":
        freq = bz.LIGHT_SPEED / np.linspace(700e-9, 400e-9, 31)
    wl = bz.LIGHT_SPEED / freq
    axis = graded_axis(args.dx, half_nm=args.padding)
    xy = (
        graded_axis(args.dx, half_nm=args.period / 2)
        if args.lateral == "periodic"
        else axis
    )
    grid = Grid(xy, xy, axis)
    source = bz.PlaneWaveSource(
        center=(0, 0, -200e-9),
        size=(xy[-1] * 2, xy[-1] * 2, 0),
        direction="+z",
        source_time=bz.GaussianPulse(f0, f0 / 5),
    )
    monitors = []
    # Two enclosing surfaces provide an independent near-field flux consistency check.
    for half in (6, 8):
        for a in range(3):
            for sign in (-1, 1):
                center = [0.0, 0.0, 0.0]
                center[a] = sign * half * 1e-9
                size = [2 * half * 1e-9] * 3
                size[a] = 0.0
                monitors.append(
                    bz.FluxMonitor(
                        center=center,
                        size=size,
                        freqs=freq,
                        name=f"face-{half}-{a}-{sign}",
                        interval=20,
                    )
                )
    monitors.append(
        bz.FieldMonitor(
            center=(0, 0, 0), size=(0, 0, 0), freqs=freq, name="incident", interval=20
        )
    )
    boundary = (
        [bz.Periodic(axes=("x", "y"))] if args.lateral == "periodic" else []
    ) + [
        bz.PML(
            edges=("front", "back") if args.lateral == "periodic" else "all",
            thickness=120e-9,
            formulation="cpml",
            alpha_max=args.cpml_alpha,
        )
    ]
    mat = material(args.material)
    ref, meta = execute(
        Scene((bz.Material(1),)), grid, [source], monitors, boundary, args.runtime
    )
    reference = {m.name: fields(ref[m.name]) for m in monitors}
    illumination_error = float(
        np.max(
            abs(
                376.730313668
                * reference["incident"]["Hy"]
                / reference["incident"]["Ex"]
                - 1
            )
        )
    )
    if illumination_error > 0.01:
        raise RuntimeError(
            f"Incident plane-wave impedance error {illumination_error:.3g} exceeds 1%"
        )
    weights = {m.name: np.asarray(ref[m.name].integration_weights) for m in monitors}
    jax.clear_caches()
    dev, device_meta = execute(
        Scene((bz.Material(1), mat), (Object(Sphere((0, 0, 0), 5e-9), 1),)),
        grid,
        [source],
        monitors,
        boundary,
        args.runtime,
    )
    incident = reference["incident"]
    intensity = 0.5 * (
        incident["Ex"] * incident["Hy"].conj() - incident["Ey"] * incident["Hx"].conj()
    ).real.mean(axis=1)
    exact_abs, exact_scat = mie(mat.eps_model(freq), wl)
    saved = dict(
        wavelength_nm=wl * 1e9,
        absorption_exact=exact_abs,
        scattering_exact=exact_scat,
        intensity=intensity,
        epsilon=mat.eps_model(freq),
    )
    report = dict(
        material=args.material,
        dx_nm=args.dx,
        runtime_fs=args.runtime,
        padding_nm=args.padding,
        period_nm=args.period,
        lateral_boundary=args.lateral,
        cpml_alpha_max=args.cpml_alpha,
        actual_period_nm=float(2 * xy[-1] * 1e9),
        actual_xy_dx_nm=float(np.min(np.diff(xy)) * 1e9),
        device=str(jax.devices()),
        **meta,
        device_execution=device_meta,
        magnetic_timing_aligned=True,
        surfaces=[],
    )
    raw = {}
    for m in monitors:
        for label, values in (
            ("reference", reference[m.name]),
            ("device", fields(dev[m.name])),
        ):
            for c, value in values.items():
                if not np.isfinite(value).all():
                    raise ValueError(f"Nonfinite {label} {m.name} {c}")
                raw[f"{label}_{m.name}_{c}"] = value
        raw[f"weights_{m.name}"] = weights[m.name]
    np.savez_compressed(out / "surface_fields.npz", **raw)
    for half in (6, 8):
        absorb = np.zeros(len(freq))
        scatter = absorb.copy()
        empty = absorb.copy()
        for a in range(3):
            for sign in (-1, 1):
                name = f"face-{half}-{a}-{sign}"
                d = fields(dev[name])
                r = reference[name]
                s = {c: d[c] - r[c] for c in d}
                absorb -= sign * (flux(d, a, weights[name]) - flux(r, a, weights[name]))
                scatter += sign * flux(s, a, weights[name])
                empty += sign * flux(r, a, weights[name])
        absorb /= intensity
        scatter /= intensity
        empty /= intensity
        saved.update(
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
                    max(abs(absorb - exact_abs)) / max(exact_abs)
                ),
                scattering_relative_max=float(
                    max(abs(scatter - exact_scat)) / max(exact_scat)
                ),
                absorption_peak_nm=float(wl[np.argmax(absorb)] * 1e9),
                exact_peak_nm=float(wl[np.argmax(exact_abs)] * 1e9),
                min_absorption=float(min(absorb)),
                min_scattering=float(min(scatter)),
            )
        )
    np.savez(out / "spectra.npz", **saved)
    np.savez(out / "grid.npz", x=xy, y=xy, z=axis)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def reflector(args):
    label = (
        f"{args.dx:g}nm"
        if args.mesh == "refined"
        else f"reference-{args.lengths[0]:g}nm"
    )
    out = OUT / f"reflector-{args.gold_model}-{label}-{args.runtime:g}fs-outgoing"
    out.mkdir(parents=True, exist_ok=True)
    dx = args.dx * 1e-9
    x = np.linspace(-60e-9, 60e-9, round(120 / args.dx) + 1)
    y = np.linspace(-150e-9, 150e-9, round(300 / args.dx) + 1)
    # Explicit interface-aligned layers, coarser homogeneous exterior.
    z = np.unique(
        np.concatenate(
            [
                np.linspace(a, b, int(np.ceil((b - a) / step)) + 1)
                for a, b, step in [
                    (-837.5e-9, -130e-9, 10e-9),
                    (-130e-9, 80e-9, dx),
                    (80e-9, 837.5e-9, 10e-9),
                ]
            ]
        )
    )
    if args.mesh == "reference":
        if len(args.lengths) != 1:
            raise ValueError(
                "Reference grid requires exactly one antenna length per run"
            )
        length = args.lengths[0] or 40
        saved_grid = np.load(DATA / f"reflector_grid_{length:g}nm.npz")
        x, y, z = (saved_grid[a] for a in "xyz")
    grid = Grid(x, y, z)
    freq = np.array([bz.LIGHT_SPEED / 850e-9])
    source = bz.PlaneWaveSource(
        center=(0, 0, 255e-9),
        size=(120e-9, 300e-9, 0),
        direction="-z",
        pol_angle=np.pi / 2,
        source_time=bz.GaussianPulse(freq[0], freq[0] / 10),
    )
    middle = int(np.argmin(abs(z - 510e-9)))
    incoming = int(np.argmin(abs(z - 180e-9)))
    plane_z = dict(
        r0=float(z[middle - 1]),
        r=float(z[middle]),
        r2=float(z[middle + 1]),
        input0=float(z[incoming]),
        input1=float(z[incoming + 1]),
        t=-300e-9,
    )
    monitors = [
        bz.FluxMonitor(
            center=(0, 0, h),
            size=(120e-9, 300e-9, 0),
            freqs=freq,
            name=n,
            interval=10,
        )
        for n, h in plane_z.items()
    ]
    boundaries = [
        bz.Periodic(axes=("x", "y")),
        bz.PML(
            edges=("front", "back"),
            thickness=None if args.mesh == "reference" else 200e-9,
            formulation="cpml",
            alpha_max=0.0,
            kappa_max=3.0 if args.mesh == "reference" else 2.0,
        ),
    ]
    ref, meta = execute(
        Scene((bz.Material(1),)),
        grid,
        [source],
        monitors,
        boundaries,
        args.runtime,
        courant=0.99 if args.mesh == "reference" else 0.8,
    )
    refs = {m.name: fields(ref[m.name]) for m in monitors}
    weights = {m.name: np.asarray(ref[m.name].integration_weights) for m in monitors}
    jax.clear_caches()
    materials = (
        bz.Material(1),
        material(
            "reflector_gold_snapshot"
            if args.gold_model == "snapshot"
            else "reflector_gold"
        ),
        bz.Material(1.892),
        bz.Material(2.25),
    )
    objects = [
        Object(Box((-60e-9, -150e-9, z[0]), (60e-9, 150e-9, -130e-9)), 3),
        Object(Box((-60e-9, -150e-9, -130e-9), (60e-9, 150e-9, 0)), 1),
        Object(Box((-60e-9, -150e-9, 0), (60e-9, 150e-9, 50e-9)), 2),
    ]
    records = []
    for length in args.lengths:
        objs = objects + (
            [
                Object(
                    Box(
                        (-45e-9, -length * 0.5e-9, 50e-9),
                        (45e-9, length * 0.5e-9, 80e-9),
                    ),
                    1,
                )
            ]
            if length
            else []
        )
        dev, device_meta = execute(
            Scene(materials, tuple(objs)),
            grid,
            [source],
            monitors,
            boundaries,
            args.runtime,
            courant=0.99 if args.mesh == "reference" else 0.8,
        )
        row = dict(length_nm=length, execution=device_meta)
        save = {}
        reference_e, device_e = {}, {}
        for name in ("r0", "r", "r2", "input0", "input1"):
            d = fields(dev[name])
            r = refs[name]
            w = weights[name]
            reference_e[name] = np.sum(r["Ey"] * w, axis=1) / w.sum()
            device_e[name] = np.sum(d["Ey"] * w, axis=1) / w.sum()
            save[f"reference_{name}"] = reference_e[name]
            save[f"device_{name}"] = device_e[name]

        def current(values, lo, hi):
            return -np.imag(values[lo] * values[hi].conj()) / (
                plane_z[hi] - plane_z[lo]
            )

        # Above the planar TF/SF source only outgoing radiation is present.
        # Recover injected incident power from empty-domain net downward plus
        # returned upward power; do not subtract the empty-grid reflection.
        incident_power = -current(reference_e, "input0", "input1") + current(
            reference_e, "r", "r2"
        )
        for name, pair in (("r", ("r", "r2")), ("r2", ("r0", "r"))):
            power = current(device_e, *pair) / incident_power
            if np.any(power < 0) or not np.isfinite(power).all():
                raise ValueError("Invalid outgoing native-plane power")
            local_phase = np.angle(device_e[name] / reference_e["input0"])
            amp = np.sqrt(power) * np.exp(1j * local_phase)
            row[name + "_real"] = float(amp[0].real)
            row[name + "_imag"] = float(amp[0].imag)
            row[name + "_power"] = float(power[0])
            save[name] = amp
        row["empty_grid_reflectivity"] = float(
            (current(reference_e, "r", "r2") / incident_power)[0]
        )
        row["transmission"] = float(dev["t"].flux[0] / ref["input0"].flux[0])
        records.append(row)
        np.savez(out / f"{length:g}nm.npz", **save)
        (out / "report.json").write_text(
            json.dumps(
                dict(
                    device=str(jax.devices()),
                    dx_nm=args.dx,
                    gold_model=args.gold_model,
                    mesh=args.mesh,
                    reflection_measurement="above_source_native_power",
                    monitor_z_nm={k: v * 1e9 for k, v in plane_z.items()},
                    runtime_fs=args.runtime,
                    **meta,
                    runs=records,
                ),
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(row), flush=True)
        jax.clear_caches()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("case", choices=["sphere", "reflector"])
    p.add_argument("--dx", type=float, default=1)
    p.add_argument("--runtime", type=float, default=90)
    p.add_argument("--padding", type=float, default=450)
    p.add_argument("--period", type=float, default=350)
    p.add_argument("--lateral", choices=["periodic", "cpml"], default="periodic")
    p.add_argument("--cpml-alpha", type=float, default=0.0)
    p.add_argument("--mesh", choices=["refined", "reference"], default="refined")
    p.add_argument("--gold-model", choices=["snapshot", "sdk"], default="snapshot")
    p.add_argument(
        "--material", choices=["sphere_gold", "aluminum"], default="sphere_gold"
    )
    p.add_argument(
        "--lengths", type=float, nargs="+", default=[40, 132.3076923077, 280]
    )
    args = p.parse_args()
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("Local GPU required")
    (sphere if args.case == "sphere" else reflector)(args)


if __name__ == "__main__":
    main()
