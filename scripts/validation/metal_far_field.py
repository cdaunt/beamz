"""Free-space radiation of closed-surface equivalent currents.

Integrate electric and magnetic currents before squaring the far-field
amplitude. For a periodic array this isolates one particle's radiating response;
the direct per-cell array flux is a different observable. Array interactions can
still perturb the reconstructed response relative to an isolated particle.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss


def directions(order=10):
    mu, weights = leggauss(order)
    phi = np.arange(2 * order) * np.pi / order
    directions = np.stack(
        np.broadcast_arrays(
            np.sqrt(1 - mu[:, None] ** 2) * np.cos(phi),
            np.sqrt(1 - mu[:, None] ** 2) * np.sin(phi),
            mu[:, None],
        ),
        axis=-1,
    ).reshape(-1, 3)
    solid = np.broadcast_to(
        weights[:, None] * np.pi / order, (order, 2 * order)
    ).ravel()
    return directions, solid


def radiation_amplitude(surfaces, k, look, impedance=376.730313668):
    """Return A with E_far = i k exp(i k r) A / (4 pi r)."""
    electric = np.zeros((len(look), 3), complex)
    magnetic = electric.copy()
    for position, normal, e, h, area in surfaces:
        phase = np.exp(-1j * k * (look @ position.T)) * area
        electric += phase @ np.cross(normal, h)
        magnetic -= phase @ np.cross(normal, e)
    transverse = electric - look * np.sum(electric * look, axis=1)[:, None]
    return impedance * transverse - np.cross(look, magnetic)


def radiated_power(surfaces, k, impedance=376.730313668):
    """Exp(-i wt) phasors; surfaces contain position, normal, E, H, area."""
    look, solid = directions()
    amplitude = radiation_amplitude(surfaces, k, look, impedance)
    return float(
        k
        * k
        / (32 * np.pi * np.pi * impedance)
        * np.sum(solid * np.sum(abs(amplitude) ** 2, axis=1))
    )


def sphere_projection(raw, grid, wavelength_nm, intensity, half):
    cells = (grid[:-1] + grid[1:]) / 2
    selected = cells[(cells > -half * 1e-9) & (cells < half * 1e-9)]
    t0, t1 = np.meshgrid(selected, selected, indexing="ij")
    surface = []
    for axis in range(3):
        tangential = ((2, 1), (2, 0), (1, 0))[axis]
        for sign in (-1, 1):
            name = f"face-{half}-{axis}-{sign}"
            position = np.zeros((t0.size, 3))
            position[:, axis] = sign * half * 1e-9
            position[:, tangential[0]] = t0.ravel()
            position[:, tangential[1]] = t1.ravel()
            normal = np.zeros(3)
            normal[axis] = sign
            components = {}
            for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                key = "device_" + name + "_" + c
                components[c] = (
                    raw[key] - raw["reference_" + name + "_" + c]
                    if key in raw
                    else np.zeros((len(wavelength_nm), t0.size), complex)
                )
            electric = np.stack([components["E" + a] for a in "xyz"], axis=-1)
            magnetic = np.stack([components["H" + a] for a in "xyz"], axis=-1)
            surface.append(
                (position, normal, electric, magnetic, raw["weights_" + name])
            )
    scattering, extinction = [], []
    for i, wl in enumerate(wavelength_nm):
        samples = [(p, n, e[i], h[i], w) for p, n, e, h, w in surface]
        k = 2 * np.pi / (wl * 1e-9)
        scattering.append(radiated_power(samples, k) / intensity[i])
        forward = radiation_amplitude(samples, k, np.array([[0.0, 0.0, 1.0]]))
        incident = raw["reference_incident_Ex"][i].mean()
        extinction.append(
            (forward[0, 0] * incident.conjugate()).real
            / (2 * 376.730313668 * intensity[i])
        )
    return np.array(scattering), np.array(extinction)


def validate_dipole():
    # Analytic dipole in dimensionless vacuum (epsilon=mu=c=Z=k=1).
    errors, extinction_errors = [], []
    for count in (20, 40):
        half = 0.2
        coordinate = (np.arange(count) + 0.5) * 2 * half / count - half
        aa, bb = np.meshgrid(coordinate, coordinate, indexing="ij")
        p = np.array([1.0 + 2.0j, 0.0, 0.0])
        surfaces = []
        for axis in range(3):
            other = [a for a in range(3) if a != axis]
            for sign in (-1, 1):
                position = np.zeros((aa.size, 3))
                position[:, axis] = sign * half
                position[:, other[0]] = aa.ravel()
                position[:, other[1]] = bb.ravel()
                r = np.linalg.norm(position, axis=1)[:, None]
                n = position / r
                e = (
                    np.exp(1j * r)
                    / (4 * np.pi)
                    * (
                        np.cross(np.cross(n, p), n) / r
                        + (3 * n * (n @ p)[:, None] - p) * (1 / r**3 - 1j / r**2)
                    )
                )
                h = np.exp(1j * r) / (4 * np.pi) * np.cross(n, p) * (1 / r + 1j / r**2)
                normal = np.zeros(3)
                normal[axis] = sign
                surfaces.append(
                    (position, normal, e, h, np.full(aa.size, (2 * half / count) ** 2))
                )
        value = radiated_power(surfaces, 1, 1)
        errors.append(abs(value / (5 / (12 * np.pi)) - 1))
        forward = radiation_amplitude(surfaces, 1, np.array([[0.0, 0.0, 1.0]]), 1)
        extinction_errors.append(float(abs(forward[0, 0].real / 2 - 1)))
    assert errors[-1] < 0.001 and errors[-1] < errors[0]
    assert (
        extinction_errors[-1] < 0.001 and extinction_errors[-1] < extinction_errors[0]
    )
    return dict(radiated_power=errors, optical_theorem=extinction_errors)


if __name__ == "__main__":
    print("Analytic dipole radiation relative errors:", validate_dipole())
