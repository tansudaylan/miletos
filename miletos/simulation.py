"""Helpers for synthetic instrument time series."""

from __future__ import annotations

import numpy as np


SOLAR_MASS_KG = 2 * 10**30  # [kg / M_sun]
SOLAR_RADIUS_M = 6.957 * 10**8  # [m / R_sun]
EARTH_RADIUS_M = 6.371e6  # [m]


def minimum_transit_inclination(radius_star_m, radius_planet_m, semimajor_axis_m):
    """Return the grazing transit inclination for lengths in meters, in degrees."""
    transit_cosine_limit = (radius_star_m + radius_planet_m) / semimajor_axis_m
    return np.degrees(np.arccos(np.clip(transit_cosine_limit, -1.0, 1.0)))


def semimajor_axis_from_period(mass_star, period_days):
    """Return semimajor axis [m] for stellar mass [M_sun] and period [day]."""
    mass_star_kg = mass_star * SOLAR_MASS_KG  # [kg]
    return semimajor_axis_from_mass_kg(mass_star_kg, period_days)


def semimajor_axis_from_mass_kg(mass_star, period_days):
    """Return semimajor axis [m] for stellar mass [kg] and period [day]."""
    period_seconds = period_days * 24 * 3600  # [s]
    return ((6.6743 * 10**-11 * mass_star * period_seconds**2) / (4 * np.pi**2)) ** (1 / 3)


def generate_batman_light_curve(radius_star, mass_star, radius_planet, albedo_planet,
                                period, inclination, time_array, a=None):
    """Return times, median-normalized uniform-disk flux, and duration placeholder.

    Radii are in solar radii, stellar mass in solar masses, period and times
    in days, inclination in degrees, and optional semimajor axis in meters.
    This retains Albus's circular, instantaneous primary-transit model with
    its first sample at mid-transit. The third return remains zero days,
    not an estimated transit duration. The historical ``albedo_planet``
    argument is assigned to BATMAN's flux-ratio parameter.
    """
    import batman

    star_radius_m = radius_star * SOLAR_RADIUS_M  # [m]
    planet_radius_m = radius_planet * SOLAR_RADIUS_M  # [m]
    time = np.array([time_array]) if isinstance(time_array, float) else np.array(time_array)
    if a is None:
        a = semimajor_axis_from_period(mass_star, period)
    params = batman.TransitParams()
    params.t0 = time if time.size == 1 else min(time)
    params.per = period
    params.rp = planet_radius_m / star_radius_m
    params.a = a / star_radius_m
    params.inc = inclination
    params.ecc = 0
    params.w = 90  # [deg]
    params.u = []
    params.fp = albedo_planet
    params.limb_dark = 'uniform'
    params.t_secondary = 0.5  # [orbital phase]
    model = batman.TransitModel(params, time, transittype='primary')
    flux = model.light_curve(params)
    flux = flux / np.median(flux)
    duration = 0  # [day]
    return time, flux, duration


def inject_batman_transit(tic_id, lc, time_array, radius_star=0.01, mass_star=0.6,
                         radius_planet=0.1, albedo_planet=0.1, period=1,
                         inclination=90, a=None, ID='', folder=''):
    """Multiply measured flux by the Albus-compatible BATMAN transit model.

    Input units follow ``generate_batman_light_curve``. Preserve Albus's
    missing injected uncertainties rather than inventing a noise model.
    A nonempty folder also saves the injected-curve plot.
    """
    from pathlib import Path

    from .light_curve import LightCurve
    import matplotlib.pyplot as plt
    from tdpy.verbosity import print

    time, flux, _ = generate_batman_light_curve(
        radius_star=radius_star, mass_star=mass_star, radius_planet=radius_planet,
        albedo_planet=albedo_planet, period=period, inclination=inclination,
        time_array=time_array, a=a,
    )
    injected = LightCurve(time=time, flux=flux * lc['flux'].value)
    if folder != '':
        injected.scatter()
        plt.xlabel('Time (days)')
        plt.ylabel('Relative Flux')
        plt.title(f'ID {ID} with TIC {tic_id} Light Curve')
        output_path = Path(folder) / f'{ID}_injectedlc.png'
        print(f'Writing to {output_path}...')
        plt.savefig(output_path)
        plt.close()
    return injected


def _log_bound_ratio(min_val, max_val):
    """Retain narrow relative widths without subtracting large logarithms."""
    with np.errstate(over='ignore', under='ignore'):
        relative_width = (max_val - min_val) / min_val
        if np.isfinite(relative_width):
            return np.log1p(relative_width)
    return np.log(max_val) - np.log(min_val)


def _stable_power_quantile(min_val, max_val, exponent, uniform):
    """Evaluate the inverse CDF in logs, anchored at the dominant endpoint."""
    log_lower, log_upper = np.log(min_val), np.log(max_val)
    log_width = _log_bound_ratio(min_val, max_val)
    with np.errstate(over='ignore', under='ignore', divide='ignore'):
        if exponent == 0:
            log_samples = (1 - uniform) * log_lower + uniform * log_upper
            anchor = min_val
            log_offset = uniform * log_width
        else:
            extent = abs(exponent) * log_width
            weight = uniform if exponent < 0 else 1 - uniform
            if extent <= 1:
                log_weight = np.log1p(weight * np.expm1(-extent))
            else:
                log_weight = np.logaddexp(np.log1p(-weight), np.log(weight) - extent)
            anchor = min_val if exponent < 0 else max_val
            log_offset = log_weight / exponent
            log_samples = (log_lower if exponent < 0 else log_upper) + log_offset
        if log_width <= 1:
            samples = anchor + anchor * np.expm1(log_offset)
        else:
            samples = np.exp(log_samples)
    samples = np.where(uniform == 0, min_val, np.where(uniform == 1, max_val, samples))
    return samples[()] if samples.ndim == 0 else samples


def sample_power_law(min_val, max_val, alpha, size=None):
    """Draw from a bounded density proportional to x**(-alpha).

    Bounds share the caller's units. NumPy's global random state and one
    uniform draw per sample are retained for seeded Albus experiments.
    """
    if not np.isfinite([min_val, max_val, alpha]).all() or not 0 < min_val <= max_val:
        raise ValueError('Power-law bounds must be finite, positive, ordered, and the slope finite')
    uniform = np.random.random(size=size)
    if min_val == max_val:
        return min_val + 0 * uniform
    if alpha == 1.0:
        with np.errstate(over='ignore'):
            ratio = max_val / min_val
        if np.isfinite(ratio) and _log_bound_ratio(min_val, max_val) > 1:
            return min_val * ratio ** uniform
        return _stable_power_quantile(min_val, max_val, 0, uniform)
    exponent = 1.0 - alpha
    with np.errstate(over='ignore', under='ignore'):
        extent = abs(exponent) * _log_bound_ratio(min_val, max_val)
    if extent <= 1:
        return _stable_power_quantile(min_val, max_val, exponent, uniform)
    try:
        with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
            lower_power = min_val**exponent
            upper_power = max_val**exponent
            if lower_power > 0 and upper_power > 0 and lower_power != upper_power and np.isfinite([lower_power, upper_power]).all():
                samples = (lower_power + (upper_power - lower_power) * uniform) ** (1.0 / exponent)
                if np.isfinite(samples).all() and np.all(samples >= min_val) and np.all(samples <= max_val):
                    return samples
    except (OverflowError, ZeroDivisionError, FloatingPointError):
        pass
    return _stable_power_quantile(min_val, max_val, exponent, uniform)


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

    from tdpy import retr_noisphot

    noise_ppt = float(np.asarray(retr_noisphot(magnitude, 'ULTRASAT')).reshape(-1)[0])
    flux_uncertainty = noise_ppt * 1e-3 * np.sqrt(3600.0 / exposure_seconds)
    flux = model_flux + np.random.default_rng(seed).normal(0.0, flux_uncertainty, time_jd.size)
    return np.column_stack((time_jd, flux, np.full(time_jd.shape, flux_uncertainty)))