"""Helpers for synthetic instrument time series."""

from __future__ import annotations

import numpy as np


def generate_ultrasat_time_series(
    start_jd: float = 2462000.0,
    duration_days: float = 1.0,
    cadence_seconds: float = 300.0,
    exposure_seconds: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Return ULTRASAT sample times [JD], cadence [s], and exposure [s]."""

    values = np.asarray((start_jd, duration_days, cadence_seconds), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("ULTRASAT start time, duration, and cadence must be finite.")
    if duration_days <= 0.0 or cadence_seconds <= 0.0:
        raise ValueError("ULTRASAT duration and cadence must be positive.")

    if exposure_seconds is None:
        exposure_seconds = 0.8 * cadence_seconds
    exposure_seconds = float(exposure_seconds)
    if not np.isfinite(exposure_seconds) or exposure_seconds <= 0.0:
        raise ValueError("ULTRASAT exposure_seconds must be finite and positive.")
    if exposure_seconds > cadence_seconds:
        raise ValueError("ULTRASAT exposure_seconds cannot exceed cadence_seconds.")

    offsets_days = np.arange(0.0, duration_days, cadence_seconds / 86400.0)
    time_jd = start_jd + offsets_days
    return time_jd, float(cadence_seconds), exposure_seconds


def simulate_ultrasat_light_curve(
    relative_flux_model,
    magnitude: float,
    start_jd: float = 2462000.0,
    duration_days: float = 1.0,
    cadence_seconds: float = 300.0,
    exposure_seconds: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate ULTRASAT photometry as columns of JD, relative flux, and uncertainty.

    ``relative_flux_model`` may be a callable evaluated at each sample time or
    an array with one relative-flux value per sample.
    """

    time_jd, _, exposure_seconds = generate_ultrasat_time_series(
        start_jd=start_jd,
        duration_days=duration_days,
        cadence_seconds=cadence_seconds,
        exposure_seconds=exposure_seconds,
    )
    model_flux = relative_flux_model(time_jd) if callable(relative_flux_model) else relative_flux_model
    model_flux = np.asarray(model_flux, dtype=float)
    if model_flux.ndim == 0:
        model_flux = np.full(time_jd.shape, model_flux.item())
    if model_flux.shape != time_jd.shape or not np.isfinite(model_flux).all():
        raise ValueError("relative_flux_model must return one finite value per sample time.")
    if np.any(model_flux <= 0.0):
        raise ValueError("relative_flux_model values must be positive.")

    from nicomedia import retr_noisphot

    noise_ppt = float(np.asarray(retr_noisphot(magnitude, 'ULTRASAT')).reshape(-1)[0])
    flux_uncertainty = noise_ppt * 1e-3 * np.sqrt(3600.0 / exposure_seconds)
    flux = model_flux + np.random.default_rng(seed).normal(0.0, flux_uncertainty, time_jd.size)
    return np.column_stack((time_jd, flux, np.full(time_jd.shape, flux_uncertainty)))