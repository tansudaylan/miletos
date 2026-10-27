"""Optional comparisons with external analysis packages, never used by workflows."""

from time import perf_counter

import numpy as np


def compare_lightkurve_bls(curve):
    """Compare native and Lightkurve BLS on the same fixed grid.

    Lightkurve is imported only when this explicit comparison is requested.
    Install it separately to run this benchmark. It is not a Miletos dependency.
    """
    import lightkurve
    from .box_least_squares import fixed_grid_box_least_squares
    from .light_curve import as_light_curve

    native_curve = as_light_curve(curve)
    reference_curve = lightkurve.LightCurve(time=native_curve.time, flux=native_curve.flux,
                                          flux_err=native_curve.flux_err)
    periods = np.logspace(0, 1, 2048)  # [day]
    start = perf_counter()
    native = fixed_grid_box_least_squares(native_curve)
    native_seconds = perf_counter() - start  # [s]
    start = perf_counter()
    reference = reference_curve.to_periodogram(method='bls', period=periods,
                                              duration=periods * 0.01, frequency_factor=5000)
    reference_seconds = perf_counter() - start  # [s]
    return {'native': native, 'lightkurve': reference, 'native_seconds': native_seconds,
            'lightkurve_seconds': reference_seconds}