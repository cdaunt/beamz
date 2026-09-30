"""Empty-domain CPML reflection as the interior mesh is refined."""

import json

import numpy as np
from metal_benchmarks import OUT, execute, fields, graded_axis

import beamz as bz
from beamz.design.raster import Grid, Scene


def main():
    rows = []
    for dx in (1, 0.5, 0.25):
        for alpha in (None, 0.0):
            z = graded_axis(dx)
            x = np.arange(5) * dx * 1e-9 - 2 * dx * 1e-9
            grid = Grid(x, x, z)
            f0 = bz.LIGHT_SPEED / 530e-9
            freq = np.linspace(0.8 * f0, 1.2 * f0, 31)
            result, meta = execute(
                Scene((bz.Material(1),)),
                grid,
                [
                    bz.PlaneWaveSource(
                        center=(0, 0, -200e-9),
                        size=(4 * dx * 1e-9, 4 * dx * 1e-9, 0),
                        direction="+z",
                        source_time=bz.GaussianPulse(f0, f0 / 5),
                    )
                ],
                [
                    bz.FieldMonitor(
                        center=(0, 0, 0),
                        size=(0, 0, 0),
                        freqs=freq,
                        name="center",
                        interval=20,
                    )
                ],
                [
                    bz.Periodic(axes=("x", "y")),
                    bz.PML(
                        edges=("front", "back"),
                        thickness=120e-9,
                        formulation="cpml",
                        alpha_max=alpha,
                    ),
                ],
                90,
            )
            field = fields(result["center"])
            e = field["Ex"][:, 0]
            h = field["Hy"][:, 0] * 376.730313668
            row = dict(
                dx_nm=dx,
                alpha_max=alpha,
                impedance_error=float(max(abs(h / e - 1))),
                reflection_amplitude=float(max(abs((e - h) / (e + h)))),
                **meta,
            )
            rows.append(row)
            print(row, flush=True)
            (OUT / "pml_incident_check.json").write_text(
                json.dumps(rows, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
