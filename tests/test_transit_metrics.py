"""Measurement conventions retained by the Albus compatibility imports."""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units

from miletos import transit_metrics as metrics


def test_fitted_parameters_retain_the_original_quantity_objects():
    periodogram = SimpleNamespace(period_at_max_power=2 * units.day,
                                 duration_at_max_power=3 * units.hour)
    assert metrics.period_at_max_power(periodogram) is periodogram.period_at_max_power
    assert metrics.duration_at_max_power(periodogram) is periodogram.duration_at_max_power


@pytest.mark.parametrize('quantity', [False, True])
def test_depth_scatter_and_snr_retain_population_scatter_convention(quantity):
    flux = np.array([1.01, 0.99, 0.97, 0.96])
    if quantity:
        flux = flux * units.dimensionless_unscaled
    mask = np.array([False, False, True, True])
    depths = metrics.transit_depth(np.arange(4), flux, mask)
    np.testing.assert_allclose(depths, [0.035, 0.035, 0.04])
    assert metrics.out_of_transit_variability(flux, mask) == np.std(flux[~mask])
    assert metrics.transit_snr(flux, mask) == depths[1] / np.std(flux[~mask])


@pytest.mark.parametrize('mask', [np.zeros(0, dtype=bool), np.zeros(3, dtype=bool), np.ones(3, dtype=bool)])
def test_undefined_measurements_keep_nan_results_without_numerical_warnings(mask):
    flux = np.ones(mask.size)
    with np.errstate(all='raise'):
        assert np.isnan(metrics.transit_snr(flux, mask))
        if mask.all():
            assert np.isnan(metrics.out_of_transit_variability(flux, mask))
        if not mask.any():
            assert np.isnan(metrics.transit_depth(np.arange(mask.size), flux, mask)).all()


def test_v_shape_retains_the_threshold_and_time_span_definition():
    time = np.arange(6, dtype=float)  # [day]
    flux = np.array([1., 0.99, 0.95, 0.95, 0.99, 1.])
    mask = np.ones(6, dtype=bool)
    assert metrics.transit_v_shape(time, flux, mask) == 2.


@pytest.mark.parametrize('phase0,nbins,expected_depth', [
    (0., 25, 0.02243085797116895),
    (0.12, 25, 0.022683046203927026),
    (0., 150, 0.022213222478886907),
    (0.12, 150, 0.022562352473065705),
])
def test_quantile_depth_retains_phase_origin_support_and_uncertainty(phase0, nbins, expected_depth):
    """Retain pre-migration outputs, including the estimator's depth bias."""
    phase = np.linspace(-0.5, 0.5, 10000)
    centered = (phase - phase0 + 0.5) % 1 - 0.5
    flux = 1 + np.random.default_rng(13).normal(0, 0.001, phase.size)
    flux[np.abs(centered) < 0.02] -= 0.02
    original = flux.copy()
    result = metrics.transit_depth_quantile_phase({'time': phase, 'flux': flux},
                                                0.04, phase0=phase0, nbins=nbins)
    assert result['depth'] == pytest.approx(expected_depth, rel=1e-12)
    assert abs(result['floor_phase'] - phase0) < 0.02
    assert result['depth_err'] == result['sigma_oot'] / np.sqrt(result['N_eff'])
    assert result['snr'] == result['depth'] / result['depth_err']
    np.testing.assert_array_equal(result['oot_mask'], np.abs(centered) > 1.2 * 0.02)
    np.testing.assert_array_equal(result['window_mask'], np.abs(centered) <= 1.5 * 0.02)
    np.testing.assert_array_equal(flux, original)


@pytest.mark.parametrize('phase,duration,message', [
    (np.linspace(-0.01, 0.01, 100), 0.1, 'Too few OOT points'),
    (np.linspace(-0.5, 0.5, 100), 0.01, 'Not enough points'),
    (np.tile([-0.5, -0.02, 0., 0.02, 0.5], 100), 0.1, 'Too few populated bins'),
])
def test_quantile_depth_retains_insufficient_sampling_errors(phase, duration, message):
    with pytest.raises(ValueError, match=message):
        metrics.transit_depth_quantile_phase({'time': phase, 'flux': np.ones(phase.size)}, duration)