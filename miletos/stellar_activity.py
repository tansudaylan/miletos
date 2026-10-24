"""Fit stellar spot-and-flare light curves with Ephesos forward models."""

import numpy as np
from scipy.optimize import least_squares

from ephesos.stellar_activity import evaluate_spot_flare_model


def parse_spot_flare_parameters(parameters, *, number_spots, number_flares=0, evolving_spots=False):
    """Unpack six stellar entries, spot blocks, then four entries per flare.

    Stellar entries are u1, u2, period [day], inclination [deg], relative shear,
    and flux offset. Spot blocks are latitude [deg], longitude [deg], and radius
    ratio, optionally followed by Gaussian peak time and width [day]. Flare
    blocks are peak time [day], amplitude, rise time [day], and decay time [day].
    """

    for count in (number_spots, number_flares):
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError("Spot and flare counts must be nonnegative integers")
    parameters = np.asarray(parameters, dtype=float)
    spot_block_size = 5 if evolving_spots else 3
    spot_end = 6 + number_spots * spot_block_size
    if parameters.shape != (spot_end + 4 * number_flares,) or not np.isfinite(parameters).all():
        raise ValueError("Parameter vector must be finite and match the specified component counts")
    spots = parameters[6:spot_end].reshape(number_spots, spot_block_size)
    flares = parameters[spot_end:].reshape(number_flares, 4)
    model_parameters = {
        "limb_darkening": parameters[:2],
        "rotation_period_days": parameters[2],
        "inclination_degrees": parameters[3],
        "relative_shear": parameters[4],
        "flux_offset": parameters[5],
        "spot_latitudes_degrees": spots[:, 0],
        "spot_longitudes_degrees": spots[:, 1],
        "spot_radius_ratios": spots[:, 2],
        "flare_peak_times_days": flares[:, 0],
        "flare_amplitudes": flares[:, 1],
        "flare_rise_times_days": flares[:, 2],
        "flare_decay_times_days": flares[:, 3],
    }
    if evolving_spots:
        model_parameters["spot_peak_times_days"] = spots[:, 3]
        model_parameters["spot_widths_days"] = spots[:, 4]
    return model_parameters


def log_prior_spot_flare(parameters, *, number_spots, number_flares=0, evolving_spots=False):
    """Preserve Hattusa's descending-radius ordering prior for interchangeable spots."""

    try:
        model_parameters = parse_spot_flare_parameters(
            parameters, number_spots=number_spots, number_flares=number_flares,
            evolving_spots=evolving_spots,
        )
    except ValueError:
        return -np.inf
    radii = model_parameters["spot_radius_ratios"]
    return 0.0 if np.all(radii[:-1] >= radii[1:]) else -np.inf


def _time_series(time_days, relative_flux, flux_uncertainty):
    if any(np.ma.is_masked(value) for value in (time_days, relative_flux, flux_uncertainty)):
        raise ValueError("Time, flux, and uncertainty must be unmasked vectors")
    arrays = tuple(np.asarray(value, dtype=float)
                   for value in (time_days, relative_flux, flux_uncertainty))
    if any(array.ndim != 1 or array.size == 0 or not np.isfinite(array).all() for array in arrays):
        raise ValueError("Time, flux, and uncertainty must be nonempty finite vectors")
    if len({array.size for array in arrays}) != 1 or np.any(arrays[2] <= 0):
        raise ValueError("Time-series lengths must match and uncertainties must be positive")
    return arrays


def log_likelihood_spot_flare(
    parameters, time_days, relative_flux, flux_uncertainty,
    *, number_spots, number_flares=0, evolving_spots=False, spot_contrast=0.7,
):
    """Evaluate the independent Gaussian likelihood, omitting parameter-independent constants."""

    time_days, relative_flux, flux_uncertainty = _time_series(
        time_days, relative_flux, flux_uncertainty,
    )
    model_parameters = parse_spot_flare_parameters(
        parameters, number_spots=number_spots, number_flares=number_flares,
        evolving_spots=evolving_spots,
    )
    model_flux = evaluate_spot_flare_model(time_days, spot_contrast=spot_contrast, **model_parameters)
    return -0.5 * np.sum(((relative_flux - model_flux) / flux_uncertainty) ** 2)


def fit_spot_flare_light_curve(
    time_days, relative_flux, flux_uncertainty, initial_parameters,
    *, number_spots, bounds, number_flares=0, evolving_spots=False, spot_contrast=0.7,
    max_nfev=500,
):
    """Fit a fixed-count model with bounded least squares, without posterior sampling.

    Supply physically admissible bounds. This is a local fit; spot geometries
    can be degenerate and the result does not establish a unique surface map.
    The ordering prior is available separately for Bayesian inference.
    """

    time_days, relative_flux, flux_uncertainty = _time_series(
        time_days, relative_flux, flux_uncertainty,
    )

    def residuals(parameters):
        model_parameters = parse_spot_flare_parameters(
            parameters, number_spots=number_spots, number_flares=number_flares,
            evolving_spots=evolving_spots,
        )
        model_flux = evaluate_spot_flare_model(time_days, spot_contrast=spot_contrast, **model_parameters)
        return (relative_flux - model_flux) / flux_uncertainty

    result = least_squares(residuals, initial_parameters, bounds=bounds, max_nfev=max_nfev)
    model_flux = relative_flux - result.fun * flux_uncertainty
    return {"optimizer": result, "parameters": result.x, "model_flux": model_flux,
            "log_likelihood": -0.5 * np.sum(result.fun**2)}


def select_observed_variable_stars(catalog, observed_sectors, first_sector=40):
    """Select catalog rows observed at or after a one-based TESS sector.

    Coverage rows must be aligned with catalog rows, and columns must represent
    consecutive sectors starting at Sector 1. This does not perform a sky match.
    """

    coverage = np.asarray(observed_sectors, dtype=float)
    if coverage.ndim != 2 or coverage.shape[0] != len(catalog):
        raise ValueError("Coverage must have one row per catalog target")
    if not np.isfinite(coverage).all() or not np.isin(coverage, [0, 1]).all():
        raise ValueError("Coverage flags must be zero or one with no missing values")
    if (isinstance(first_sector, (bool, np.bool_))
            or not isinstance(first_sector, (int, np.integer))
            or not 1 <= first_sector <= coverage.shape[1]):
        raise ValueError("first_sector must identify a supplied one-based sector column")
    observed = coverage[:, first_sector - 1:].any(axis=1)
    return catalog.iloc[np.flatnonzero(observed)].copy()