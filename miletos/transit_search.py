"""Single-transit, repeating-transit, and TTV-guided moon searches."""

from __future__ import annotations

import numpy as np

from .box_least_squares import search_box_least_squares


def _validate_curve(time_days, flux, flux_uncertainty):
    time_days = np.asarray(time_days, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_uncertainty = np.asarray(flux_uncertainty, dtype=float)
    if (time_days.ndim != 1 or time_days.size < 3 or flux.shape != time_days.shape
            or flux_uncertainty.shape != time_days.shape):
        raise ValueError("time_days, flux, and flux_uncertainty must be matching vectors")
    if not all(np.isfinite(values).all() for values in (time_days, flux, flux_uncertainty)):
        raise ValueError("light-curve arrays must be finite")
    if np.any(np.diff(time_days) <= 0.0) or np.any(flux_uncertainty <= 0.0):
        raise ValueError("times must increase strictly and uncertainties must be positive")
    return time_days, flux, flux_uncertainty


def search_repeating_transits(
    time_days,
    flux,
    flux_uncertainty,
    *,
    duration_days=(0.04, 0.07, 0.10, 0.14, 0.20),
    minimum_period_days: float = 0.5,
    maximum_period_days: float = 20.0,
    maximum_candidates: int = 6,
    minimum_depth_snr: float = 7.0,
    minimum_transits: int = 3,
) -> list[dict[str, float | int]]:
    """Iteratively search periodic dips with Miletos BLS and mask recovered trains."""

    time_days, flux, flux_uncertainty = _validate_curve(time_days, flux, flux_uncertainty)
    if maximum_candidates < 1 or minimum_transits < 2:
        raise ValueError("candidate and transit-count limits must be positive")
    available = np.ones(time_days.size, dtype=bool)
    candidates = []
    for _ in range(maximum_candidates):
        if available.sum() < 3:
            break
        result = search_box_least_squares(
            time_days[available],
            flux[available],
            flux_uncertainty[available],
            duration_days=duration_days,
            minimum_period_days=minimum_period_days,
            maximum_period_days=maximum_period_days,
        )
        period = float(result["period_days"])
        epoch = float(result["transit_time_days"])
        duration = float(result["duration_days"])
        phase = np.remainder(time_days - epoch + 0.5 * period, period) - 0.5 * period
        in_transit = available & (np.abs(phase) <= 0.5 * duration)
        event_numbers = np.rint((time_days[in_transit] - epoch) / period).astype(int)
        event_count = int(np.unique(event_numbers).size)
        if result["depth_snr"] < minimum_depth_snr or event_count < minimum_transits:
            break
        if any(abs(period / candidate["period_days"] - 1.0) < 0.01 for candidate in candidates):
            break
        candidates.append({
            "period_days": period,
            "transit_time_days": epoch,
            "duration_days": duration,
            "depth": float(result["depth"]),
            "depth_uncertainty": float(result["depth_uncertainty"]),
            "depth_snr": float(result["depth_snr"]),
            "transit_count": event_count,
        })
        available &= np.abs(phase) > 0.75 * duration
    return candidates


def search_single_transit_events(
    time_days,
    flux,
    flux_uncertainty,
    *,
    duration_days=(0.04, 0.07, 0.10, 0.14, 0.20, 0.30),
    baseline_half_window_days: float = 0.5,
    excluded_intervals=(),
    maximum_candidates: int = 10,
    minimum_in_transit_points: int = 3,
) -> list[dict[str, float | int]]:
    """Rank isolated box dips using weighted local in/out means.

    ``excluded_intervals`` contains ``(start_day, end_day)`` pairs, for example
    the repeated planetary transit windows found by ``search_repeating_transits``.
    Returned SNRs are screening statistics and are not a multiple-trial false-alarm
    probability.
    """

    time_days, flux, flux_uncertainty = _validate_curve(time_days, flux, flux_uncertainty)
    durations = np.atleast_1d(np.asarray(duration_days, dtype=float))
    if (not np.isfinite(durations).all() or np.any(durations <= 0.0)
            or baseline_half_window_days <= 0.0 or maximum_candidates < 1):
        raise ValueError("search durations, baseline window, and candidate count must be positive")

    usable = np.ones(time_days.size, dtype=bool)
    intervals = np.asarray(excluded_intervals, dtype=float)
    if intervals.size:
        if intervals.ndim != 2 or intervals.shape[1] != 2 or not np.isfinite(intervals).all():
            raise ValueError("excluded_intervals must be finite (start, end) pairs")
        if np.any(intervals[:, 1] <= intervals[:, 0]):
            raise ValueError("excluded intervals must have positive width")
        for start_day, end_day in intervals:
            usable &= (time_days < start_day) | (time_days > end_day)

    search_time = time_days[usable]
    search_flux = flux[usable]
    search_error = flux_uncertainty[usable]
    weights = 1.0 / search_error**2
    cumulative_weight = np.concatenate(([0.0], np.cumsum(weights)))
    cumulative_flux = np.concatenate(([0.0], np.cumsum(weights * search_flux)))
    all_candidates = []

    for duration in durations:
        centers = search_time
        in_left = np.searchsorted(search_time, centers - 0.5 * duration, side="left")
        in_right = np.searchsorted(search_time, centers + 0.5 * duration, side="right")
        local_left = np.searchsorted(
            search_time, centers - baseline_half_window_days, side="left"
        )
        local_right = np.searchsorted(
            search_time, centers + baseline_half_window_days, side="right"
        )
        in_weight = cumulative_weight[in_right] - cumulative_weight[in_left]
        in_sum = cumulative_flux[in_right] - cumulative_flux[in_left]
        local_weight = cumulative_weight[local_right] - cumulative_weight[local_left]
        local_sum = cumulative_flux[local_right] - cumulative_flux[local_left]
        out_weight = local_weight - in_weight
        out_sum = local_sum - in_sum
        in_count = in_right - in_left
        out_count = (local_right - local_left) - in_count
        valid = ((in_count >= minimum_in_transit_points) & (out_count >= 3)
                 & (in_weight > 0.0) & (out_weight > 0.0))
        depth = np.full(centers.size, np.nan)
        depth_uncertainty = np.full(centers.size, np.nan)
        depth[valid] = out_sum[valid] / out_weight[valid] - in_sum[valid] / in_weight[valid]
        depth_uncertainty[valid] = np.sqrt(1.0 / in_weight[valid] + 1.0 / out_weight[valid])
        snr = depth / depth_uncertainty
        for index in np.flatnonzero(valid & np.isfinite(snr) & (depth > 0.0)):
            all_candidates.append({
                "transit_time_days": float(centers[index]),
                "duration_days": float(duration),
                "depth": float(depth[index]),
                "depth_uncertainty": float(depth_uncertainty[index]),
                "depth_snr": float(snr[index]),
                "in_transit_points": int(in_count[index]),
            })

    all_candidates.sort(key=lambda candidate: candidate["depth_snr"], reverse=True)
    selected = []
    for candidate in all_candidates:
        if all(
            abs(candidate["transit_time_days"] - previous["transit_time_days"])
            > 0.5 * (candidate["duration_days"] + previous["duration_days"])
            for previous in selected
        ):
            selected.append(candidate)
            if len(selected) == maximum_candidates:
                break
    return selected


def fit_periodic_ttv(
    epochs,
    ttv_days,
    uncertainty_days,
    *,
    minimum_cycle_period: float = 2.0,
    maximum_cycle_period: float | None = None,
    grid_size: int = 500,
) -> dict[str, object]:
    """Fit a weighted sinusoidal TTV model over a period grid measured in epochs."""

    epochs = np.asarray(epochs, dtype=float)
    ttv_days = np.asarray(ttv_days, dtype=float)
    uncertainty_days = np.asarray(uncertainty_days, dtype=float)
    if (epochs.ndim != 1 or epochs.size < 5 or ttv_days.shape != epochs.shape
            or uncertainty_days.shape != epochs.shape):
        raise ValueError("TTV inputs must be matching vectors with at least five epochs")
    if (not all(np.isfinite(values).all() for values in (epochs, ttv_days, uncertainty_days))
            or np.any(uncertainty_days <= 0.0)):
        raise ValueError("TTV values must be finite and uncertainties positive")
    if maximum_cycle_period is None:
        maximum_cycle_period = max(minimum_cycle_period + 0.1, epochs.size / 2.0)
    if not minimum_cycle_period < maximum_cycle_period or grid_size < 20:
        raise ValueError("TTV period bounds or grid_size are invalid")

    weights = 1.0 / uncertainty_days**2
    offset = float(np.sum(weights * ttv_days) / np.sum(weights))
    chi_squared_constant = float(np.sum(weights * (ttv_days - offset) ** 2))
    periods = np.linspace(minimum_cycle_period, maximum_cycle_period, grid_size)
    best = None
    for cycle_period in periods:
        angle = 2.0 * np.pi * (epochs - epochs[0]) / cycle_period
        design = np.column_stack((np.ones(epochs.size), np.sin(angle), np.cos(angle)))
        coefficients = np.linalg.lstsq(
            design * np.sqrt(weights[:, None]),
            ttv_days * np.sqrt(weights),
            rcond=None,
        )[0]
        residual = ttv_days - design @ coefficients
        chi_squared = float(np.sum(weights * residual**2))
        if best is None or chi_squared < best["chi_squared"]:
            best = {
                "cycle_period": float(cycle_period),
                "coefficients": coefficients,
                "chi_squared": chi_squared,
            }

    amplitude = float(np.hypot(best["coefficients"][1], best["coefficients"][2]))
    angle = 2.0 * np.pi * (epochs - epochs[0]) / best["cycle_period"]
    fitted = (best["coefficients"][0]
              + best["coefficients"][1] * np.sin(angle)
              + best["coefficients"][2] * np.cos(angle))
    delta_bic = chi_squared_constant - best["chi_squared"] - 3.0 * np.log(epochs.size)
    return {
        "cycle_period": best["cycle_period"],
        "amplitude_days": amplitude,
        "amplitude_minutes": amplitude * 1440.0,
        "offset_days": float(best["coefficients"][0]),
        "fitted_ttv_days": fitted,
        "chi_squared_constant": chi_squared_constant,
        "chi_squared_sinusoid": best["chi_squared"],
        "delta_bic": float(delta_bic),
        "epochs": epochs,
    }


def search_transit_origami(
    time_days,
    flux,
    flux_uncertainty,
    *,
    linear_transit_times_days,
    fitted_ttv_days,
    observed_transit_times_days,
    transit_duration_days: float,
    mass_ratio_grid,
    excluded_intervals=(),
) -> list[dict[str, float | int]]:
    """Search Kipping's TTV-guided transit-origami moon windows.

    For each trial satellite-to-planet mass ratio, the moon timing is predicted by
    ``TTV_moon = -TTV_planet / mass_ratio``. Planet transit windows and optional
    contaminating transit intervals are excluded before the weighted box-vs-flat
    comparison. This is a screening statistic, not a photodynamic validation.
    """

    time_days, flux, flux_uncertainty = _validate_curve(time_days, flux, flux_uncertainty)
    linear_times = np.asarray(linear_transit_times_days, dtype=float)
    fitted_ttv = np.asarray(fitted_ttv_days, dtype=float)
    observed_times = np.asarray(observed_transit_times_days, dtype=float)
    mass_ratios = np.asarray(mass_ratio_grid, dtype=float)
    if (linear_times.ndim != 1 or linear_times.size < 3
            or not all(values.shape == linear_times.shape
                       for values in (fitted_ttv, observed_times))):
        raise ValueError("transit timing arrays must be matching vectors with at least three events")
    if (not all(np.isfinite(values).all() for values in
                (linear_times, fitted_ttv, observed_times, mass_ratios))
            or np.any(mass_ratios <= 0.0) or transit_duration_days <= 0.0):
        raise ValueError("origami inputs must be finite with positive durations and mass ratios")

    usable = np.ones(time_days.size, dtype=bool)
    for center in observed_times:
        usable &= np.abs(time_days - center) > 0.75 * transit_duration_days
    intervals = np.asarray(excluded_intervals, dtype=float)
    if intervals.size:
        if intervals.ndim != 2 or intervals.shape[1] != 2 or not np.isfinite(intervals).all():
            raise ValueError("excluded_intervals must be finite (start, end) pairs")
        for start_day, end_day in intervals:
            usable &= (time_days < start_day) | (time_days > end_day)

    weights = 1.0 / flux_uncertainty**2
    rows = []
    for mass_ratio in mass_ratios:
        moon_times = linear_times - fitted_ttv / mass_ratio
        moon_window = np.zeros(time_days.size, dtype=bool)
        local_window = np.zeros(time_days.size, dtype=bool)
        for moon_time in moon_times:
            separation = np.abs(time_days - moon_time)
            moon_window |= separation <= 0.5 * transit_duration_days
            local_window |= separation <= 3.0 * transit_duration_days
        in_moon = usable & moon_window
        out_moon = usable & local_window & ~moon_window
        if in_moon.sum() < 3 or out_moon.sum() < 3:
            continue
        weight_in = float(np.sum(weights[in_moon]))
        weight_out = float(np.sum(weights[out_moon]))
        mean_in = float(np.sum(weights[in_moon] * flux[in_moon]) / weight_in)
        mean_out = float(np.sum(weights[out_moon] * flux[out_moon]) / weight_out)
        depth = mean_out - mean_in
        depth_uncertainty = np.sqrt(1.0 / weight_in + 1.0 / weight_out)
        signed_snr = float(depth / depth_uncertainty)
        rows.append({
            "mass_ratio": float(mass_ratio),
            "moon_depth": float(depth),
            "depth_uncertainty": float(depth_uncertainty),
            "signed_snr": signed_snr,
            "delta_chi_squared": float(np.sign(depth) * signed_snr**2),
            "in_moon_points": int(in_moon.sum()),
            "out_moon_points": int(out_moon.sum()),
            "epochs_in_fold": int(np.count_nonzero([
                np.any(in_moon & (np.abs(time_days - moon_time) <= 0.5 * transit_duration_days))
                for moon_time in moon_times
            ])),
        })
    return rows
