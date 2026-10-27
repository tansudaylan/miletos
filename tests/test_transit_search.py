import numpy as np
import pytest

from miletos.transit_search import (
    fit_periodic_ttv,
    search_repeating_transits,
    search_single_transit_events,
    search_transit_origami,
    transit_mask,
)


@pytest.mark.parametrize('include_edges', [False, True])
def test_transit_mask_preserves_day_conversion_and_boundary_convention(include_edges):
    from astropy import units
    from astropy.time import Time

    offsets = np.array([0., 0.125, 0.25, 0.5, 0.75, 0.875, 1.]) * units.day
    epoch = Time('2026-10-26', scale='tdb')
    expected = [True, True, include_edges, False, include_edges, True, True]
    for time, origin in ((offsets.to(units.s), 0 * units.hour), (epoch + offsets, epoch)):
        np.testing.assert_array_equal(
            transit_mask(time, 24 * units.hour, origin, 12 * units.hour, include_edges=include_edges),
            expected,
        )


def test_single_transit_search_recovers_an_isolated_box():
    time_days = np.arange(0.0, 20.0, 0.02)
    injected_epoch = 9.37
    injected_duration = 0.10
    flux = np.ones_like(time_days)
    flux[np.abs(time_days - injected_epoch) < injected_duration / 2.0] -= 0.01
    flux += np.random.default_rng(9).normal(0.0, 0.001, time_days.size)
    uncertainty = np.full(time_days.size, 0.001)

    candidates = search_single_transit_events(
        time_days,
        flux,
        uncertainty,
        duration_days=[0.08, 0.10, 0.12],
        excluded_intervals=[(4.0, 5.0)],
    )

    assert candidates
    assert candidates[0]["transit_time_days"] == pytest.approx(injected_epoch, abs=0.03)
    assert candidates[0]["depth"] == pytest.approx(0.01, abs=0.002)
    assert candidates[0]["depth_snr"] > 10.0


def test_repeating_transit_search_uses_miletos_bls():
    time_days = np.arange(0.0, 30.0, 0.02)
    period_days = 2.5
    phase_days = np.remainder(time_days - 0.4 + period_days / 2.0, period_days) - period_days / 2.0
    flux = np.ones_like(time_days)
    flux[np.abs(phase_days) < 0.06] -= 0.01
    flux += np.random.default_rng(10).normal(0.0, 0.001, time_days.size)

    candidates = search_repeating_transits(
        time_days,
        flux,
        np.full(time_days.size, 0.001),
        duration_days=[0.10, 0.12, 0.14],
        minimum_period_days=2.0,
        maximum_period_days=3.0,
        maximum_candidates=1,
    )

    assert len(candidates) == 1
    assert candidates[0]["period_days"] == pytest.approx(period_days, abs=0.02)
    assert candidates[0]["transit_count"] >= 3


def test_ttv_sinusoid_and_origami_grid_recover_injected_mass_ratio():
    epochs = np.arange(12, dtype=float)
    cycle_period = 4.0
    ttv_days = 0.01 * np.sin(2.0 * np.pi * epochs / cycle_period + 0.3)
    ttv_uncertainty_days = np.full(epochs.size, 0.001)
    fit = fit_periodic_ttv(
        epochs,
        ttv_days,
        ttv_uncertainty_days,
        minimum_cycle_period=2.0,
        maximum_cycle_period=6.0,
        grid_size=500,
    )
    assert fit["cycle_period"] == pytest.approx(cycle_period, abs=0.03)
    assert fit["delta_bic"] > 10.0

    linear_times = 5.0 + 2.0 * epochs
    observed_times = linear_times + ttv_days
    true_mass_ratio = 0.05
    moon_times = linear_times - ttv_days / true_mass_ratio
    time_days = np.arange(0.0, 30.0, 0.01)
    flux = np.ones_like(time_days)
    for event_time in observed_times:
        flux[np.abs(time_days - event_time) <= 0.06] -= 0.01
    for event_time in moon_times:
        flux[np.abs(time_days - event_time) <= 0.06] -= 0.005
    flux += np.random.default_rng(11).normal(0.0, 0.001, time_days.size)

    mass_ratio_grid = np.geomspace(0.02, 0.12, 100)
    origami = search_transit_origami(
        time_days,
        flux,
        np.full(time_days.size, 0.001),
        linear_transit_times_days=linear_times,
        fitted_ttv_days=ttv_days,
        observed_transit_times_days=observed_times,
        transit_duration_days=0.12,
        mass_ratio_grid=mass_ratio_grid,
    )
    best = max(origami, key=lambda row: row["delta_chi_squared"])

    assert best["mass_ratio"] == pytest.approx(true_mass_ratio, rel=0.08)
    assert best["moon_depth"] > 0.0
