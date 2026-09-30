"""Run with tidy3d==2.9.0; no cloud calls."""

import json
from pathlib import Path

import numpy as np
import tidy3d as td


def main():
    if td.__version__ != "2.9.0":
        raise RuntimeError("Use tidy3d==2.9.0 to reproduce the saved notebook grid")
    out = Path(__file__).resolve().parents[2] / "examples/data/metal_benchmarks"
    s = td.Simulation.from_file(str(out / "tidy3d_reflector_snapshot.json"))
    rows = []
    for length in [40, 132.3076923076923, 280]:
        objects = list(s.structures)
        objects[-1] = objects[-1].updated_copy(
            geometry=td.Box.from_bounds(
                rmin=(-0.045, -length * 0.0005, 0.05),
                rmax=(0.045, length * 0.0005, 0.08),
            )
        )
        sim = s.updated_copy(structures=objects)
        data = {
            axis: np.asarray(getattr(sim.grid.boundaries, axis)) * 1e-6
            for axis in "xyz"
        }
        np.savez(out / f"reflector_grid_{length:g}nm.npz", **data)
        if length == 132.3076923076923:
            samples = {
                component: np.asarray(
                    sim.epsilon_on_grid(grid=sim.grid, coord_key=component, freq=np.inf)
                ).transpose(2, 1, 0)
                for component in ("Ex", "Ey", "Ez")
            }
            np.savez_compressed(out / "reflector_point_epsilon.npz", **samples)
        rows.append(dict(length_nm=length, grid=list(sim.grid.num_cells), dt=sim.dt))
    (out / "reference_grids.json").write_text(
        json.dumps(dict(version=td.__version__, runs=rows), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
