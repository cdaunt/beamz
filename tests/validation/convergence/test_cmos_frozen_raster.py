"""The memory-bounded CMOS diagnostic must preserve the dense constitutive grid."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from beamz.design.discretization import MaterialGrid
from beamz.design.raster import RasterOptions
from beamz.design.raster.dispersion import rasterize_dispersion


def test_cmos_frozen_slab_coefficients_agree_with_whole_geometry(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "scripts/validation"))
    monkeypatch.syspath_prepend(str(root / "examples"))
    from cmos_frequency_frozen import freeze_dielectrics
    from cmos_frozen_raster import frozen_raster
    from cmos_reference_mesh import reference_grid, reference_scene

    # Preserve every reference z interface; slabs cut across multiple materials.
    grid = reference_grid(200e-9)
    scene = reference_scene(grid)
    frequency, dt = 299792458 / 450e-9, 1.4413465606999474e-17
    options = RasterOptions(periodic_axes=("x", "y"), reference_frequency=frequency)
    dense = MaterialGrid.from_raster_result(
        scene.rasterize(grid, options=options), dimensions=3
    )
    regions, interfaces = rasterize_dispersion(
        scene,
        grid,
        kind="3d",
        polarization="tm",
        quality=options.quality,
        resolution=grid.minimum_spacing,
        cache_directory=None,
        smoothing=options.smoothing,
        reference_frequency=frequency,
        periodic_axes=("x", "y"),
    )
    dense, _ = freeze_dielectrics(
        replace(dense, dispersion=regions, dispersion_interfaces=interfaces),
        frequency,
        dt,
    )
    chunked = frozen_raster(scene, grid, frequency, dt, slab_cells=9)
    assert float(chunked.permeability) == 1
    assert float(chunked.conductivity) == 0
    for name in ("mu_hx", "mu_hy", "mu_hz"):
        np.testing.assert_array_equal(dense.yee_materials[name], 1)
    for name in chunked.yee_materials:
        np.testing.assert_allclose(
            chunked.yee_materials[name],
            dense.yee_materials[name],
            rtol=2e-6,
            atol=1e-7,
            err_msg=name,
        )
    assert len(dense.dispersion) == len(chunked.dispersion) == 1
    for (ma, wa), (mb, wb) in zip(dense.dispersion, chunked.dispersion, strict=True):
        assert ma == mb
        for component in wa:
            np.testing.assert_allclose(wa[component], wb[component], atol=1e-7)
