"""Field images must not mix raw DFT sums with physical incident amplitudes."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from examples.cmos_rgb_sensor import plot_rgb_fields


def test_cmos_field_images_use_weight_normalized_components():
    shape = (3, 4)
    physical = dict(Ex=np.full(shape, 2.0), Ey=np.zeros(shape), Ez=np.zeros(shape))
    # Raw sums contain the acquisition weight; the public accessor removes it.
    monitor = SimpleNamespace(
        dft_fields={c: 1000 * v for c, v in physical.items()},
        get_dft_component=lambda component: physical[component],
        sample_region=SimpleNamespace(axis_interval=lambda axis: slice(0, 2)),
    )

    class Results(dict):
        metadata = SimpleNamespace(
            grid=SimpleNamespace(axis_edges=lambda axis: np.arange(3) * 1e-6),
            coordinate_offset=(0, 0, 0),
        )

    result = Results(
        dict.fromkeys(("field_xy_silicon", "field_yz_BG", "field_yz_GR"), monitor)
    )
    figures = plot_rgb_fields(result, {}, incident_intensity=np.full(3, 4.0))
    try:
        for figure in figures:
            for axis in figure.axes[:3]:
                np.testing.assert_allclose(axis.collections[0].get_array(), 1.0)
    finally:
        for figure in figures:
            plt.close(figure)
