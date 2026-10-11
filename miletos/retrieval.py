"""PCAT-backed fixed-dimensional spectral retrievals for JWST data."""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np


def _spectral_log_likelihood(parameters, state):
    """Evaluate a covariance-aware Gaussian likelihood for a binned spectrum."""
    model_error_types = (ArithmeticError, ValueError) + state["model_error_types"]
    try:
        model_depth_ppm = np.asarray(
            state["forward_model"](np.asarray(parameters, dtype=float), state["wavelength_microns"]),
            dtype=float,
        )
    except model_error_types:
        return -np.inf
    if model_depth_ppm.shape != state["observed_depth_ppm"].shape:
        raise ValueError("forward_model must return one transit depth per wavelength bin")
    if not np.isfinite(model_depth_ppm).all():
        return -np.inf
    residual_ppm = state["observed_depth_ppm"] - model_depth_ppm
    whitened_residual = np.linalg.solve(state["covariance_cholesky"], residual_ppm)
    return float(-0.5 * np.dot(whitened_residual, whitened_residual))


def sample_nirspec_retrieval(
    wavelength_microns: Sequence[float],
    observed_depth_ppm: Sequence[float],
    covariance_ppm2: np.ndarray,
    forward_model: Callable[[np.ndarray, np.ndarray], Sequence[float]],
    parameter_names: Sequence[str],
    prior_types: Sequence[str],
    prior_minima: Sequence[float],
    prior_maxima: Sequence[float],
    initial_values: Sequence[float],
    proposal_scales: Sequence[float],
    *,
    prior_means: Sequence[float] | None = None,
    prior_stdvs: Sequence[float] | None = None,
    model_error_types: Sequence[type[Exception]] = (),
    chain_count: int = 4,
    retained_samples: int = 10000,
    burn_samples: int = 10000,
    seed: int | None = None,
    pathbase: str | Path | None = None,
    typeverb: int = 0,
    **sampler_options,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a binned NIRSpec retrieval posterior with PCAT's fixed-chain sampler.

    ``forward_model(parameters, wavelength_microns)`` must return model transit
    depths in ppm at the observed bin centers. It may wrap a radiative-transfer
    code and include parameters such as ``log10_hdo_h2o`` or another isotopic
    abundance. Miletos handles the correlated Gaussian likelihood and delegates
    all posterior sampling to PCAT; isotope-specific opacities must be supplied
    by the forward model.

    Returns the chain in ``(chain, retained_sample, parameter)`` order and the
    corresponding log-posterior array in ``(chain, retained_sample)`` order.

    """
    from pcat.fixed import sample_fixed_chains

    wavelength_microns = np.asarray(wavelength_microns, dtype=float)
    observed_depth_ppm = np.asarray(observed_depth_ppm, dtype=float)
    covariance_ppm2 = np.asarray(covariance_ppm2, dtype=float)
    parameter_names = tuple(parameter_names)
    prior_types = tuple(prior_types)
    prior_minima = np.asarray(prior_minima, dtype=float)
    prior_maxima = np.asarray(prior_maxima, dtype=float)
    initial_values = np.asarray(initial_values, dtype=float)
    proposal_scales = np.asarray(proposal_scales, dtype=float)
    parameter_count = len(parameter_names)

    if wavelength_microns.ndim != 1 or wavelength_microns.size < 2:
        raise ValueError("wavelength_microns must be a one-dimensional array with at least two bins")
    if observed_depth_ppm.shape != wavelength_microns.shape:
        raise ValueError("observed_depth_ppm must match wavelength_microns")
    if not np.isfinite(wavelength_microns).all() or np.any(np.diff(wavelength_microns) <= 0.0):
        raise ValueError("wavelength_microns must be finite and strictly increasing")
    if not np.isfinite(observed_depth_ppm).all():
        raise ValueError("observed_depth_ppm must be finite")
    if covariance_ppm2.shape != (wavelength_microns.size, wavelength_microns.size):
        raise ValueError("covariance_ppm2 must be square with one row per wavelength bin")
    if not np.isfinite(covariance_ppm2).all() or not np.allclose(
        covariance_ppm2, covariance_ppm2.T, rtol=1e-10, atol=1e-12
    ):
        raise ValueError("covariance_ppm2 must be finite and symmetric")
    try:
        covariance_cholesky = np.linalg.cholesky(covariance_ppm2)
    except np.linalg.LinAlgError as error:
        raise ValueError("covariance_ppm2 must be positive definite") from error

    if not parameter_count or len(set(parameter_names)) != parameter_count:
        raise ValueError("parameter_names must be nonempty and unique")
    vectors = {
        "prior_types": prior_types,
        "prior_minima": prior_minima,
        "prior_maxima": prior_maxima,
        "initial_values": initial_values,
        "proposal_scales": proposal_scales,
    }
    if any(len(values) != parameter_count for values in vectors.values()):
        raise ValueError("prior and initial-value arrays must match parameter_names")
    if not np.isfinite(prior_minima).all() or not np.isfinite(prior_maxima).all():
        raise ValueError("prior bounds must be finite")
    if np.any(prior_maxima <= prior_minima):
        raise ValueError("every prior maximum must exceed its minimum")
    if not np.isfinite(initial_values).all() or not np.isfinite(proposal_scales).all():
        raise ValueError("initial_values and proposal_scales must be finite")
    if np.any(proposal_scales <= 0.0):
        raise ValueError("proposal_scales must be positive")
    if "gaus" in prior_types and (prior_means is None or prior_stdvs is None):
        raise ValueError("Gaussian prior types require prior_means and prior_stdvs")
    if prior_means is not None and len(prior_means) != parameter_count:
        raise ValueError("prior_means must match parameter_names")
    if prior_stdvs is not None and len(prior_stdvs) != parameter_count:
        raise ValueError("prior_stdvs must match parameter_names")
    if chain_count < 1 or retained_samples < 1 or burn_samples < 0:
        raise ValueError("chain_count and retained_samples must be positive; burn_samples cannot be negative")
    if not callable(forward_model):
        raise TypeError("forward_model must be callable")
    model_error_types = tuple(model_error_types)
    if not all(
        isinstance(error_type, type) and issubclass(error_type, Exception)
        for error_type in model_error_types
    ):
        raise ValueError("model_error_types must contain exception classes")

    state = {
        "wavelength_microns": wavelength_microns,
        "observed_depth_ppm": observed_depth_ppm,
        "covariance_cholesky": covariance_cholesky,
        "forward_model": forward_model,
        "model_error_types": model_error_types,
    }
    options = {
        "boolmakeplot": False,
        "boolmakeplotinit": False,
        "boolmakeplotfram": False,
        "booldiag": False,
        "typeverb": typeverb,
    }
    options.update(sampler_options)
    return sample_fixed_chains(
        state,
        _spectral_log_likelihood,
        None,
        parameter_names,
        prior_types,
        prior_minima,
        prior_maxima,
        prior_means,
        prior_stdvs,
        initial_values[None, :],
        chain_count,
        retained_samples,
        burn_samples,
        pathbase=pathbase,
        seed=seed,
        **options,
    )