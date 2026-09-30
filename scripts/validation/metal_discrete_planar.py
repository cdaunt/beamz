"""Independent 1D Yee-frequency control of empty-grid subtraction bias.

Uses transparent outgoing leads, so the exterior result need not exactly match
the time-domain CPML/source implementation. Requires no GPU or Tidy3D.
"""

import json

import numpy as np
from metal_benchmarks import DATA, OUT, material
from scipy.linalg import solve_banded

import beamz as bz
from beamz.design.raster import Box, Grid, Object, RasterOptions, Scene


def solve(z, eps, dt):
    w = 2 * np.sin(np.pi * bz.LIGHT_SPEED / 850e-9 * dt) / dt
    d = np.diff(z)
    left = np.r_[d[0], d]
    right = np.r_[d, d[-1]]
    dual = (left + right) / 2
    diag = -1 / left - 1 / right + (w / bz.LIGHT_SPEED) ** 2 * eps * dual
    ql = np.exp(2j * np.arcsin(w / bz.LIGHT_SPEED * d[0] * np.sqrt(eps[0]) / 2))
    qr = np.exp(2j * np.arcsin(w / bz.LIGHT_SPEED * d[-1] * np.sqrt(eps[-1]) / 2))
    diag[0] += ql / left[0]
    diag[-1] += qr / right[-1]
    ab = np.zeros((3, len(z)), complex)
    ab[0, 1:] = 1 / right[:-1]
    ab[1] = diag
    ab[2, :-1] = 1 / left[1:]
    rhs = np.zeros(len(z), complex)
    rhs[-1] = (qr - 1 / qr) / right[-1]
    return solve_banded((1, 1), ab, rhs)


def main():
    rows = []
    for spacing in ("reference", 5, 2.5):
        if spacing == "reference":
            saved = np.load(DATA / "reflector_grid_40nm.npz")
            x, y, z = (saved[a] for a in "xyz")
            courant = 0.99
        else:
            dx = spacing * 1e-9
            x = np.linspace(-60e-9, 60e-9, round(120 / spacing) + 1)
            y = np.linspace(-150e-9, 150e-9, round(300 / spacing) + 1)
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
            courant = 0.8
        g = Grid(x, y, z)
        m = material("reflector_gold_snapshot")
        scene = Scene(
            (bz.Material(1), m, bz.Material(1.892), bz.Material(2.25)),
            (
                Object(Box((-60e-9, -150e-9, z[0]), (60e-9, 150e-9, -130e-9)), 3),
                Object(Box((-60e-9, -150e-9, -130e-9), (60e-9, 150e-9, 0)), 1),
                Object(Box((-60e-9, -150e-9, 0), (60e-9, 150e-9, 50e-9)), 2),
            ),
        )
        sim = bz.Simulation(
            scene=scene,
            raster_grid=g,
            raster_options=RasterOptions(reference_frequency=bz.LIGHT_SPEED / 850e-9),
            boundaries=[bz.Periodic(axes=("x", "y"))],
            time=np.arange(2) * 1e-18,
        )
        mg = sim.material_grid
        dt = g.cfl_time_step(courant)
        f = np.tan(np.pi * bz.LIGHT_SPEED / 850e-9 * dt) / (np.pi * dt)
        eps = np.asarray(mg.yee_materials["eps_y"]).astype(complex)
        for medium, weights in mg.dispersion:
            eps += np.asarray(weights["Ey"]) * (
                medium.eps_model(f) - medium.epsilon_inf
            )
        eps = eps.mean(axis=(1, 2))
        dev = solve(z, eps, dt)
        ref = solve(z, np.ones(len(z), complex), dt)
        ix = np.argmin(abs(z - 180e-9))
        scat = dev - ref
        R = -np.imag(scat[ix] * scat[ix + 1].conj()) / np.imag(
            ref[ix] * ref[ix + 1].conj()
        )
        row = dict(
            spacing=spacing,
            discrete_R=float(R),
            exterior_R=float(abs(dev[-1] - 1) ** 2),
            empty_R=float(abs(ref[-1] - 1) ** 2),
        )
        rows.append(row)
        print(row)
    (OUT / "discrete_planar_checks.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
