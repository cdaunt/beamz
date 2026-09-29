"""Match the published solver log against two offline input reconstructions.

Requires the reference snapshot's Tidy3D 2.12.0.dev3; no cloud/API calls.
"""

import json
from pathlib import Path

import tidy3d as td

ROOT = Path(__file__).resolve().parents[2]


def main():
    path = ROOT / "examples/data/cmos_rgb/tidy3d_reference_simulation.json"
    version = json.loads(path.read_text())["version"]
    if td.__version__ != version:
        raise RuntimeError(f"Use snapshot SDK {version}, got {td.__version__}")
    snapshot = td.Simulation.from_file(path)
    modified = snapshot.updated_copy(
        structures=[
            obj.updated_copy(medium=td.material_library["Al"]["Rakic1995"])
            if obj.name == "metal shield" or obj.name.startswith("interconnect")
            else obj
            for obj in snapshot.structures
        ]
    )
    report = {
        name: {"grid": simulation.grid.num_cells, "dt": simulation.dt}
        for name, simulation in (
            ("viewer_snapshot", snapshot),
            ("code_library_aluminum", modified),
        )
    }
    report["published_log"] = {"grid": [390, 390, 430], "dt": 1.4413e-17}
    report["sdk_version"] = version
    output = ROOT / "docs/reviews/cmos-reverse-engineering/grid_fingerprint.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
