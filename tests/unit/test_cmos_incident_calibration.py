"""The CMOS incident reference separates incident and reflected native waves."""

import numpy as np
import pytest

import beamz as bz
from examples.cmos_reference_mesh import decompose_normal_air_wave, reference_grid


def test_native_plane_wave_decomposition_recovers_both_directions():
    f = np.array([4.5e14, 5.5e14, 7e14])
    dt = 1.44e-17
    h = 55e-9
    distance = 2 * h
    # Derive the phase independently from cos(k h), the discrete Yee relation.
    k = np.arccos(1 - 2 * (h * np.sin(np.pi * f * dt) / (bz.LIGHT_SPEED * dt)) ** 2) / h
    incident = np.array([1 + 0.3j, 0.8 - 0.2j, 2 + 1j])
    reflected = np.array([0.1j, 0.02 - 0.1j, 0])
    e0 = incident + reflected
    e1 = incident * np.exp(-1j * k * distance) + reflected * np.exp(1j * k * distance)
    got, back, _ = decompose_normal_air_wave(
        e0, e1, f, dt=dt, spacing=h, separation=distance
    )
    np.testing.assert_allclose(got, incident, atol=1e-13)
    np.testing.assert_allclose(back, reflected, atol=1e-13)


@pytest.mark.parametrize(
    "dt,spacing,separation,f",
    [
        (-1, 1, 1, [1]),
        (1, 1, 0, [1]),
        (1, 1, 1, [np.nan]),
        (1, 1, 1, [0.6]),
        (1e-18, 1e-6, 1e-7, [7e14]),
        (1e-18, 1e-9, 1e-20, [5e14]),
    ],
)
def test_invalid_or_degenerate_calibration_is_rejected(dt, spacing, separation, f):
    with pytest.raises(ValueError):
        decompose_normal_air_wave(
            [1], [1], f, dt=dt, spacing=spacing, separation=separation
        )


def test_targeted_depth_control_preserves_external_grid_and_cfl():
    original = reference_grid(20e-9)
    refined = reference_grid(20e-9, max_optical_dz=20e-9)
    for a, b in zip(original.edges[:2], refined.edges[:2], strict=True):
        np.testing.assert_array_equal(a, b)
    z, zr = original.z_edges, refined.z_edges
    np.testing.assert_array_equal(zr[np.searchsorted(zr, z)], z)
    # Silicon, both external absorbers, and the native air calibration region
    # retain exactly the original coordinates. Only the optical stack changes.
    np.testing.assert_array_equal(zr[zr < -1.554749e-6], z[z < -1.554749e-6])
    np.testing.assert_array_equal(zr[zr > 2.15e-6], z[z > 2.15e-6])
    active = (zr[:-1] >= -1.554748e-6) & (zr[1:] <= 2.054748e-6)
    assert np.max(np.diff(zr)[active]) <= 20e-9 * (1 + 1e-12)
    assert original.cfl_time_step(0.7) == refined.cfl_time_step(0.7)
    with pytest.raises(ValueError, match="finite and positive"):
        reference_grid(20e-9, max_optical_dz=0)
