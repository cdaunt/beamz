"""Fixed-mesh dielectric averaging control; metals remain staircased.

Run with JAX_PLATFORMS=cpu while the GPU is busy, or cuda for a GPU repeat.
This compares arithmetic volume averaging to the diagonal baseline; it is not
an implementation of Tidy3D's proprietary dielectric operator.
"""

import json
from functools import partial
from types import SimpleNamespace

import jax
import metal_benchmarks as benchmark


def main():
    platform = jax.devices()[0].platform
    benchmark.OUT = benchmark.OUT / "diagnostics" / f"dielectric_volume_{platform}"
    benchmark.execute = partial(benchmark.execute, smoothing="volume")
    benchmark.reflector(
        SimpleNamespace(
            dx=7.5,
            runtime=300,
            mesh="reference",
            gold_model="snapshot",
            lengths=[132.3076923076923],
        )
    )
    path = next(benchmark.OUT.glob("reflector-*/report.json"))
    report = json.loads(path.read_text())
    report["dielectric_smoothing"] = "volume"
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
