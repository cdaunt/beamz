"""The CMOS incident reference separates incident and reflected native waves."""

import numpy as np
import pytest

import beamz as bz
from examples.cmos_reference_mesh import decompose_normal_air_wave


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
