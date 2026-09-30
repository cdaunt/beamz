"""Slab-wise construction of the single-frequency CMOS diagnostic raster.

One-cell halos preserve every native Yee support at internal slab boundaries.
Only the final float32 coefficients survive each slab, limiting setup memory.
"""

import gc
import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from cmos_frequency_frozen import freeze_dielectrics

from beamz.design.discretization import MaterialGrid
from beamz.design.grid import RectilinearGrid
from beamz.design.raster import Grid, RasterOptions
from beamz.design.raster.dispersion import rasterize_dispersion
from beamz.lattice import component_shapes


def binary_staircase_weights(values):
    """Discard only sub-1e-10 raster roundoff, below the ADE active-site cutoff."""
    values = np.asarray(values)
    if np.any((abs(values) > 1e-10) & (abs(values - 1) > 1e-10)):
        raise ValueError("Compact metal supports require binary staircasing")
    return values > 0.5


def cache_path(scene, grid, frequency, dt, directory):
    digest = hashlib.sha256(scene.to_json().encode())
    for a in grid.edges:
        digest.update(a.tobytes())
    digest.update(np.array([frequency, dt]).tobytes())
    digest.update(inspect.getsource(freeze_dielectrics).encode())
    root = Path(__file__).resolve().parents[2]
    for path in sorted(
        [
            *root.glob("rust/fdtd-raster-core/src/**/*.rs"),
            root / "beamz/design/raster/dispersion.py",
            Path(__file__),
        ]
    ):
        digest.update(path.read_bytes())
    return Path(directory) / digest.hexdigest()


def frozen_raster(scene, grid, frequency, dt, *, slab_cells=16, cache_directory=None):
    if cache_directory is not None:
        cache = cache_path(scene, grid, frequency, dt, cache_directory)
        if (cache / "metadata.json").exists():
            print("Loading cached frozen coefficients", flush=True)
            metadata = json.loads((cache / "metadata.json").read_text())

            def load(name):
                return np.load(cache / (name + ".npy"), mmap_mode="r")

            return MaterialGrid(
                permittivity=load("permittivity"),
                conductivity=0.0,
                permeability=1.0,
                resolution=grid.minimum_spacing,
                shape=grid.shape_zyx,
                grid=RectilinearGrid(*grid.edges),
                yee_materials={
                    k: load(k) for k in metadata["yee_keys"] if not k.startswith("mu_")
                },
                smoothing="farjadpour_diagonal",
                dispersion=tuple(
                    (
                        scene.materials[i],
                        {
                            c: binary_staircase_weights(load(f"metal{j}_{c}"))
                            for c in ("Ex", "Ey", "Ez")
                        },
                    )
                    for j, i in enumerate(metadata["metal_indices"])
                ),
            )
        result = frozen_raster(scene, grid, frequency, dt, slab_cells=slab_cells)
        cache.mkdir(parents=True, exist_ok=True)
        for k in ("permittivity", "conductivity", "permeability"):
            np.save(cache / (k + ".npy"), getattr(result, k))
        for k, v in result.yee_materials.items():
            np.save(cache / (k + ".npy"), v)
        indices = []
        for j, (m, weights) in enumerate(result.dispersion):
            indices.append(next(i for i, v in enumerate(scene.materials) if v is m))
            for c, v in weights.items():
                np.save(cache / f"metal{j}_{c}.npy", v)
        (cache / "metadata.json").write_text(
            json.dumps(dict(yee_keys=list(result.yee_materials), metal_indices=indices))
        )
        return result

    shape = grid.shape_zyx
    shapes = component_shapes(shape)
    cells = {k: np.empty(shape, dtype=np.float32) for k in ("permittivity",)}
    coefficients = {}
    for c in ("Ex", "Ey", "Ez"):
        for prefix in ("eps_", "sig_"):
            coefficients[prefix + c[-1].lower()] = np.empty(shapes[c], dtype=np.float32)
    metals = [
        m
        for m in scene.materials
        if hasattr(m, "eps_model") and m.eps_model(frequency).real < 1
    ]
    supports = [
        {c: np.zeros(shapes[c], dtype=bool) for c in ("Ex", "Ey", "Ez")} for _ in metals
    ]
    options = RasterOptions(periodic_axes=("x", "y"), reference_frequency=frequency)
    z = grid.z_edges
    for lo in range(0, shape[0], slab_cells):
        hi = min(lo + slab_cells, shape[0])
        halo_lo, halo_hi = max(0, lo - 1), min(shape[0], hi + 1)
        subgrid = Grid(grid.x_edges, grid.y_edges, z[halo_lo : halo_hi + 1])
        base = MaterialGrid.from_raster_result(
            scene.rasterize(subgrid, options=options), dimensions=3
        )
        regions, interfaces = rasterize_dispersion(
            scene,
            subgrid,
            kind="3d",
            polarization="tm",
            quality=options.quality,
            resolution=subgrid.minimum_spacing,
            cache_directory=None,
            smoothing=options.smoothing,
            metal_smoothing=options.metal_smoothing,
            reference_frequency=frequency,
            periodic_axes=("x", "y"),
        )
        frozen, _ = freeze_dielectrics(
            replace(base, dispersion=regions, dispersion_interfaces=interfaces),
            frequency,
            dt,
        )
        if np.any(np.asarray(frozen.conductivity) != 0) or np.any(
            np.asarray(frozen.permeability) != 1
        ):
            raise ValueError(
                "Compact CMOS raster requires zero cell conductivity and unit permeability"
            )
        start = lo - halo_lo
        for k in cells:
            cells[k][lo:hi] = np.asarray(getattr(frozen, k))[start : start + hi - lo]
        for k in coefficients:
            count = hi - lo + int(coefficients[k].shape[0] == shape[0] + 1)
            coefficients[k][lo : lo + count] = frozen.yee_materials[k][
                start : start + count
            ]
        for i, (_, weights) in enumerate(frozen.dispersion):
            for c in weights:
                count = hi - lo + int(shapes[c][0] == shape[0] + 1)
                values = weights[c][start : start + count]
                supports[i][c][lo : lo + count] = binary_staircase_weights(values)
        print(f"Frozen raster slab {lo}:{hi}/{shape[0]}", flush=True)
        del base, regions, interfaces, frozen
        gc.collect()
    return MaterialGrid(
        **cells,
        conductivity=0.0,
        permeability=1.0,
        resolution=grid.minimum_spacing,
        shape=shape,
        yee_materials=coefficients,
        grid=RectilinearGrid(*grid.edges),
        origin=tuple(a[0] for a in grid.edges),
        smoothing=options.smoothing,
        dispersion=tuple(zip(metals, supports, strict=True)),
    )
