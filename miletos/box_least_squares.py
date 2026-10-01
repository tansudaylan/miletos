"""Box Least Squares searches for periodic transit-like signals."""

import numpy as np
from astropy.timeseries import BoxLeastSquares


def search_box_least_squares(
    time_days,
    flux,
    flux_uncertainty,
    duration_days,
    minimum_period_days,
    maximum_period_days,
    frequency_factor=1.0,
) -> dict[str, object]:
    """Search a sorted light curve and return the highest-SNR transit candidate."""

    time_days = np.asarray(time_days, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_uncertainty = np.asarray(flux_uncertainty, dtype=float)
    duration_days = np.atleast_1d(np.asarray(duration_days, dtype=float))
    if time_days.ndim != 1 or time_days.size < 3:
        raise ValueError("time_days must be a one-dimensional array with at least three points")
    if flux.shape != time_days.shape or flux_uncertainty.shape != time_days.shape:
        raise ValueError("flux and flux_uncertainty must match time_days")
    if not all(np.isfinite(values).all() for values in (time_days, flux, flux_uncertainty)):
        raise ValueError("time, flux, and uncertainty values must be finite")
    if np.any(np.diff(time_days) <= 0.0):
        raise ValueError("time_days must be strictly increasing")
    if np.any(flux_uncertainty <= 0.0):
        raise ValueError("flux_uncertainty must be positive")
    if not np.isfinite(duration_days).all() or np.any(duration_days <= 0.0):
        raise ValueError("duration_days must contain finite positive values")
    if not 0.0 < minimum_period_days < maximum_period_days:
        raise ValueError("period bounds must be positive and increasing")
    if np.any(duration_days >= minimum_period_days):
        raise ValueError("transit durations must be shorter than the minimum period")
    if not np.isfinite(frequency_factor) or frequency_factor <= 0.0:
        raise ValueError("frequency_factor must be finite and positive")

    model = BoxLeastSquares(time_days, flux, dy=flux_uncertainty)
    periodogram = model.autopower(
        duration_days,
        objective="snr",
        minimum_period=minimum_period_days,
        maximum_period=maximum_period_days,
        minimum_n_transit=3,
        frequency_factor=frequency_factor,
    )
    best_index = int(np.nanargmax(periodogram.power))
    return {
        "period_days": float(periodogram.period[best_index]),
        "transit_time_days": float(periodogram.transit_time[best_index]),
        "duration_days": float(periodogram.duration[best_index]),
        "depth": float(periodogram.depth[best_index]),
        "depth_uncertainty": float(periodogram.depth_err[best_index]),
        "depth_snr": float(periodogram.depth_snr[best_index]),
        "periodogram": periodogram,
        "model": model,
    }