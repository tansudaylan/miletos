import numpy as np
import pytest

from miletos.retrieval import sample_nirspec_retrieval


def hdo_injection_model(parameters, wavelength_microns):
    """Return a synthetic NIRSpec spectrum with a tunable HDO-shaped feature."""
    depth_offset_ppm, log10_hdo_h2o = parameters
    water_feature = 320.0 * np.exp(-0.5 * ((wavelength_microns - 4.30) / 0.08) ** 2)
    hdo_feature = 120.0 * 10.0 ** (log10_hdo_h2o + 5.0) * np.exp(
        -0.5 * ((wavelength_microns - 4.12) / 0.035) ** 2
    )
    return 20000.0 + depth_offset_ppm + water_feature + hdo_feature


def test_nirspec_hdo_parameter_is_sampled_through_pcat(tmp_path):
    wavelength_microns = np.linspace(3.9, 4.55, 52)
    injected_parameters = np.array([12.0, -5.0])
    uncertainty_ppm = 12.0
    random = np.random.default_rng(188)
    observed_depth_ppm = hdo_injection_model(injected_parameters, wavelength_microns)
    observed_depth_ppm += random.normal(0.0, uncertainty_ppm, wavelength_microns.size)

    chain, log_probability = sample_nirspec_retrieval(
        wavelength_microns,
        observed_depth_ppm,
        np.diag(np.full(wavelength_microns.size, uncertainty_ppm**2)),
        hdo_injection_model,
        parameter_names=("depth_offset_ppm", "log10_hdo_h2o"),
        prior_types=("self", "self"),
        prior_minima=(-80.0, -7.0),
        prior_maxima=(80.0, -3.0),
        initial_values=(0.0, -5.5),
        proposal_scales=(0.08, 0.04),
        chain_count=1,
        retained_samples=500,
        burn_samples=500,
        seed=188,
        pathbase=tmp_path,
        booladaptstdp=False,
    )

    assert chain.shape == (1, 500, 2)
    assert log_probability.shape == (1, 500)
    assert np.isfinite(log_probability).all()
    assert np.median(chain[0, :, 1]) == pytest.approx(-5.0, abs=0.25)