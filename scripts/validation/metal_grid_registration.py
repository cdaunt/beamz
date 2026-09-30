"""Translate the physical resonator relative to the fixed Yee lattice.

Continuum reflectivity is translation invariant for this laterally periodic
stack. This tests metal boundary placement, independently of mesh refinement.
"""

import argparse
from dataclasses import replace
from types import SimpleNamespace

import jax
import metal_benchmarks as benchmark
import numpy as np

from beamz.design.raster import Box


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", type=float, default=5)
    parser.add_argument("--z-shift", type=float, default=0, help="nm")
    parser.add_argument("--xy-shift", type=float, default=0, help="nm on both axes")
    args = parser.parse_args()
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("Local GPU required")
    benchmark.OUT /= f"registration/xy{args.xy_shift:g}-z{args.z_shift:g}nm"
    original = benchmark.execute

    def execute(scene, grid, *positional, **keywords):
        objects = []
        for i, obj in enumerate(scene.objects):
            lo, hi = np.array(obj.geometry.minimum), np.array(obj.geometry.maximum)
            lo[2] += args.z_shift * 1e-9
            hi[2] += args.z_shift * 1e-9
            if i == 0:
                # The substrate still fills the entire lower PML.
                lo[2] = grid.edges[2][0]
            if i == 3:
                lo[:2] += args.xy_shift * 1e-9
                hi[:2] += args.xy_shift * 1e-9
            objects.append(replace(obj, geometry=Box(tuple(lo), tuple(hi))))
        shifted = replace(scene, objects=tuple(objects))
        result, meta = original(shifted, grid, *positional, **keywords)
        meta.update(z_shift_nm=args.z_shift, xy_shift_nm=args.xy_shift)
        return result, meta

    benchmark.execute = execute
    benchmark.reflector(
        SimpleNamespace(
            dx=args.dx,
            runtime=300,
            mesh="refined",
            gold_model="snapshot",
            lengths=[132.3076923076923],
        )
    )


if __name__ == "__main__":
    main()
