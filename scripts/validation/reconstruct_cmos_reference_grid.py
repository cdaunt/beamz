"""Reconstruct the public snapshot mesh locally; never submits a cloud job.

Run: uv run --no-project --with tidy3d==2.12.0.dev3 python
     scripts/validation/reconstruct_cmos_reference_grid.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import tidy3d as td

root = Path(__file__).resolve().parents[2]
p = root / "examples/data/cmos_rgb/tidy3d_reference_simulation.json"
snapshot = json.loads(p.read_text())
if td.__version__ != snapshot["version"]:
    raise RuntimeError(
        "Install the snapshot's exact Tidy3D version to reconstruct its mesh"
    )
s = td.Simulation.from_file(p)
g = s.grid
print("version", td.__version__, "grid", g.num_cells, "dt", s.dt, flush=True)
print("pml", s.pml_thicknesses, "bounds", s.simulation_bounds, flush=True)
for axis in "xyz":
    a = np.asarray(getattr(g.boundaries, axis))
    print(
        axis, len(a), a.min(), a.max(), np.diff(a).min(), np.diff(a).max(), flush=True
    )
np.savez(
    root / "examples/data/cmos_rgb/tidy3d_reconstructed_grid.npz",
    **{a: np.asarray(getattr(g.boundaries, a)) * 1e-6 for a in "xyz"},
)
(root / "examples/data/cmos_rgb/tidy3d_reconstructed_grid.json").write_text(
    json.dumps(
        dict(
            snapshot_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            local_tidy3d_version=td.__version__,
            snapshot_version=json.loads(p.read_text())["version"],
            num_cells=g.num_cells,
            dt=s.dt,
            pml_thickness_um=s.pml_thicknesses,
            description="Locally reconstructed grid from saved simulation; not an exported server-result mesh.",
        ),
        indent=2,
    )
    + "\n"
)
