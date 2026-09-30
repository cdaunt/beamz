"""Single-frequency constitutive cross-check of the broadband CMOS ADE.

Evaluate the existing diagonal effective complex permittivity at one frequency.
Represent positive-real sites with epsilon/conductivity, while retaining the
original metal poles and binary ownership. This is a diagnostic, not a new
broadband material model. It removes dielectric polarization history storage.
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

import beamz as bz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
from cmos_rgb_sensor import make_sim, pixel_signs  # noqa: E402


def trim_host_allocator():
    """Return freed glibc heap pages during large Linux diagnostic compilations."""
    import ctypes
    import threading

    libc = ctypes.CDLL(None)
    if not hasattr(libc, "malloc_trim"):
        raise RuntimeError("Host allocator trimming requires glibc")
    libc.malloc_trim.argtypes = [ctypes.c_size_t]
    libc.malloc_trim.restype = ctypes.c_int

    def release():
        while True:
            libc.malloc_trim(0)
            time.sleep(2)

    threading.Thread(target=release, daemon=True).start()


def freeze_dielectrics(mg, frequency, dt):
    # Match the actual trapezoidal ADE response at the requested discrete frequency.
    warped = np.tan(np.pi * frequency * dt) / (np.pi * dt)
    eps = {
        "E" + a: np.array(mg.yee_materials["eps_" + a], dtype=complex) for a in "xyz"
    }
    keep = []
    for medium, supports in mg.dispersion:
        if medium.eps_model(frequency).real < 1:
            keep.append((medium, supports))
            continue
        chi = medium.eps_model(warped) - medium.epsilon_inf
        for component, weights in supports.items():
            eps[component] += chi * weights
    for interface in mg.dispersion_interfaces:
        constituents = np.array(
            [
                m.eps_model(warped) if hasattr(m, "eps_model") else m.permittivity
                for m in interface.materials
            ]
        )
        instantaneous = np.array([m.permittivity for m in interface.materials])
        harmonic = 1 / np.sum(interface.fractions / constituents[:, None], axis=0)
        harmonic_inf = 1 / np.sum(interface.fractions / instantaneous[:, None], axis=0)
        np.add.at(
            eps[interface.component].reshape(-1),
            interface.indices,
            interface.normal_squared * (harmonic - harmonic_inf),
        )
    yee = dict(mg.yee_materials)
    audit = {}
    for component, values in eps.items():
        if np.min(values.real) <= 0 or np.min(values.imag) < -1e-9:
            raise ValueError(
                "Frozen dielectric coefficients must remain passive and positive-real"
            )
        a = component[-1].lower()
        # Conductivity uses the same trapezoidal time-centering as the ADE.
        yee["eps_" + a] = values.real.astype(np.float32)
        yee["sig_" + a] = (
            2 * np.pi * warped * bz.EPS_0 * np.maximum(values.imag, 0)
        ).astype(np.float32)
        audit[component] = dict(
            min_real=float(values.real.min()), max_imag=float(values.imag.max())
        )
    return replace(
        mg, yee_materials=yee, dispersion=tuple(keep), dispersion_interfaces=()
    ), audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavelength", type=float, default=450)
    parser.add_argument("--dx", type=float, default=20)
    parser.add_argument("--runtime", type=float, default=510)
    parser.add_argument("--reference-xy", action="store_true")
    parser.add_argument("--chunked", action="store_true")
    parser.add_argument("--closed-metal-faces", action="store_true")
    parser.add_argument("--single-pass", action="store_true")
    parser.add_argument("--trim-host-allocator", action="store_true")
    args = parser.parse_args()
    if args.dx <= 0 or args.runtime != 510 or not 400 <= args.wavelength <= 700:
        raise ValueError(
            "Require positive spacing, 400–700 nm, and the original 510 fs calibration"
        )
    if args.reference_xy or args.chunked:
        jax.config.update("jax_use_simplified_jaxpr_constants", True)
    if args.trim_host_allocator:
        trim_host_allocator()
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("RTX3090 required")
    out = (
        ROOT
        / f"docs/reviews/cmos-parity-followup/frozen-{args.wavelength:g}nm-dx{args.dx:g}nm"
    )
    if args.reference_xy:
        out = out.with_name(out.name + "-reference-xy")
    elif args.chunked:
        out = out.with_name(out.name + "-chunked")
    if args.closed_metal_faces:
        out = out.with_name(out.name + "-closed-metal-faces")
    out.mkdir(parents=True, exist_ok=True)
    f = bz.LIGHT_SPEED / (args.wavelength * 1e-9)
    sim = make_sim(
        dx=200e-9 if args.reference_xy or args.chunked else args.dx * 1e-9,
        run_time=args.runtime * 1e-15,
        num_freqs=200,
        fields=False,
        reference_z=True,
        colocation_side="plus",
        reference=args.reference_xy or args.chunked,
    )
    incident_metadata = json.loads(
        (ROOT / "docs/reviews/cmos-isolation/incident/report.json").read_text()
    )
    if sim.dt != incident_metadata["dt"] or sim.num_steps != incident_metadata["steps"]:
        raise ValueError(
            "The saved incident calibration must match the simulation clock"
        )
    if args.reference_xy or args.chunked:
        from cmos_frozen_raster import frozen_raster
        from cmos_reference_mesh import reference_grid, reference_scene

        from beamz.design.raster import Grid

        if args.reference_xy:
            saved_grid = np.load(
                ROOT / "examples/data/cmos_rgb/tidy3d_reconstructed_grid.npz"
            )
            grid = Grid(*[saved_grid[a] for a in "xyz"])
        else:
            grid = reference_grid(args.dx * 1e-9)
        if sim.dt > grid.cfl_time_step(0.995):
            raise ValueError("Reference clock exceeds the fine-grid stability limit")
        mg = frozen_raster(
            reference_scene(grid),
            grid,
            f,
            sim.dt,
            cache_directory=Path.home() / ".cache/beamz-cmos-frozen",
        )
        audit = {
            "method": "slab raster with one-cell halos",
            "reference_xy": args.reference_xy,
        }
    else:
        mg, audit = freeze_dielectrics(sim.material_grid, f, sim.dt)
    if args.closed_metal_faces:
        from cmos_metal_face_control import close_metal_faces
        from cmos_reference_mesh import reference_scene

        mg, face_audit = close_metal_faces(mg, reference_scene(mg.grid), f, sim.dt)
        audit["closed_metal_faces"] = face_audit
    monitors = [
        m.updated_copy(
            freqs=[f],
            center=tuple(
                a - b for a, b in zip(m.center, sim.coordinate_offset, strict=True)
            ),
        )
        for m in sim.monitors
        if m.name.startswith("flux_si_") or m.name == "incident_plane"
    ]
    sim = sim.updated_copy(material_grid=mg, monitors=monitors)
    print(
        json.dumps(dict(grid=sim.grid.shape, dt=sim.dt, dielectric_audit=audit)),
        flush=True,
    )
    start = time.monotonic()
    first = sim.advance(
        num_steps=sim.num_steps if args.single_pass else int(300e-15 / sim.dt),
        backend="jax",
        progress=False,
        donate_state=True,
    )
    checkpoint = dict(
        elapsed_s=time.monotonic() - start,
        time_fs=float(first.state.current_step) * sim.dt * 1e15,
        fluxes={p: -float(first.results["flux_si_" + p].flux[0]) for p in pixel_signs},
    )
    if args.single_pass:
        result = first.results
    else:
        (out / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2) + "\n")
        print("Checkpoint", json.dumps(checkpoint), flush=True)
        result = sim.advance(
            state=first.state, backend="jax", progress=False, donate_state=True
        ).results
    # Same source, z mesh, timestep and duration as the existing incident calibration.
    calibration = ROOT / "docs/reviews/cmos-isolation/incident/spectra.npz"
    saved = np.load(calibration)
    wl = saved["wavelength_nm"]
    order = np.argsort(wl)
    incident = np.interp(args.wavelength, wl[order], saved["modal_power"][order])
    fluxes = {p: -float(result["flux_si_" + p].flux[0]) for p in pixel_signs}
    efficiencies = {p: 100 * value / incident for p, value in fluxes.items()}
    efficiencies["green"] = efficiencies["green1"] + efficiencies["green2"]
    report = dict(
        wavelength_nm=args.wavelength,
        grid=list(sim.grid.shape),
        reference_xy=args.reference_xy,
        dx_nm=args.dx,
        runtime_fs=args.runtime,
        elapsed_s=time.monotonic() - start,
        device=str(jax.devices()),
        dielectric_audit=audit,
        efficiency_percent=efficiencies,
        fluxes=fluxes,
        single_pass=args.single_pass,
        trim_host_allocator=args.trim_host_allocator,
        checkpoint_to_final_max_change_pp=None
        if args.single_pass
        else max(
            abs(fluxes[p] - checkpoint["fluxes"][p]) * 100 / incident
            for p in pixel_signs
        ),
        incident_power=float(incident),
        dt=sim.dt,
        simplified_jax_constants=bool(jax.config.jax_use_simplified_jaxpr_constants),
        persistent_compilation_cache=bool(jax.config.jax_enable_compilation_cache),
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
