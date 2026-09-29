"""Sampling a continuous magnetic field with a material-induced derivative jump."""

from types import SimpleNamespace

import numpy as np
import pytest

import beamz as bz
from beamz.lattice import (
    compile_yee_plane_quadrature_3d,
    component_coordinates_rectilinear,
    component_shape_3d,
    one_sided_quadratic_plan,
)


@pytest.mark.parametrize("side", ["minus", "plus"])
@pytest.mark.parametrize("graded", [False, True])
def test_one_sided_reconstruction_does_not_cross_derivative_jump(side, graded):
    x = np.array([-3, -1.5, -0.4, 0.6, 1.4, 3.1]) if graded else np.arange(-3, 3) + 0.5
    # Piecewise quadratic, continuous at the monitor but with a derivative jump.
    h = np.where(x < 0, 2 + 4 * x + 0.2 * x * x, 2 - 3 * x + 0.7 * x * x)
    i, w = one_sided_quadratic_plan(x, [0], side)
    np.testing.assert_allclose(np.sum(h[i] * w, axis=1), [2], atol=5e-7)
    assert np.all(x[i] < 0) if side == "minus" else np.all(x[i] > 0)


@pytest.mark.parametrize("side,target", [("plus", 2), ("minus", -2)])
def test_reconstruction_rejects_insufficient_side_samples(side, target):
    with pytest.raises(ValueError, match="Three Yee samples"):
        one_sided_quadratic_plan([-2, -1, 0, 1, 2], [target], side)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("side", ["minus", "plus"])
def test_plane_quadrature_reconstructs_h_without_moving_e(axis, side):
    edges = np.array([0, 0.5, 1.1, 1.8, 2.5, 3.4, 4.5, 5.8, 7.3]) * 1e-8
    grid = bz.RectilinearGrid(edges, edges, edges)
    shape = (8, 8, 8)
    center = [2.2e-8] * 3
    size = [2e-8] * 3
    size["xyz".index(axis)] = 0
    region = SimpleNamespace(axis_interval=lambda a: slice(2, 4))
    names = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    kwargs = dict(
        center=center,
        size=size,
        normal_axis=axis,
        region=region,
        resolution=1e-8,
        grid_shape=shape,
        component_shapes={c: component_shape_3d(c, shape) for c in names},
        grid=grid,
    )
    original = compile_yee_plane_quadrature_3d(**kwargs)
    changed = compile_yee_plane_quadrature_3d(**kwargs, colocation_side=side)
    for c in names[:3]:
        for a, b in zip(original.plan(c), changed.plan(c), strict=True):
            np.testing.assert_array_equal(a, b)
    hname = next(c for c in names[3:] if c[-1].lower() != axis)
    coords = component_coordinates_rectilinear(hname, grid)
    zz, yy, xx = np.meshgrid(coords["z"], coords["y"], coords["x"], indexing="ij")
    t = {"x": xx, "y": yy, "z": zz}[axis] / 1e-8 - 2.2
    h = np.where(t < 0, 2 + 3 * t + 0.4 * t * t, 2 - 4 * t + 0.2 * t * t)
    idx, w = changed.plan(hname)
    np.testing.assert_allclose((h.reshape(-1)[idx] * w).sum(axis=-1), 2, atol=1e-6)


def test_monitor_side_is_validated_immutable_and_part_of_identity():
    a = bz.FluxMonitor(center=(0, 0, 0), size=(1, 1, 0), freqs=[1])
    b = a.updated_copy(colocation_side="plus")
    assert a != b and hash(a) != hash(b)
    assert a.colocation_side is None
    with pytest.raises(ValueError, match="colocation_side"):
        a.updated_copy(colocation_side="invalid")


def test_two_dimensional_monitor_rejects_one_sided_plane_sampling():
    sim = bz.Simulation(
        size=(1e-6, 1e-6),
        resolution=100e-9,
        run_time=1e-15,
        monitors=[
            bz.FluxMonitor(
                center=(0, 0),
                size=(0, 0.5e-6, 1e-6),
                freqs=[5e14],
                colocation_side="plus",
            )
        ],
    )
    with pytest.raises(ValueError, match="requires a 3D plane"):
        sim.compile(backend="jax")


@pytest.mark.parametrize(
    "x,t,side",
    [
        ([0, 1], [0.5], "plus"),
        ([0, 2, 1], [0.5], "plus"),
        ([0, 1, 2], [np.nan], "plus"),
        ([0, 1, 2], [0.5], "wrong"),
    ],
)
def test_invalid_reconstruction_coordinates(x, t, side):
    with pytest.raises(ValueError):
        one_sided_quadratic_plan(x, t, side)
