import numpy as np
import pytest

import miletos
from miletos.simulation import generate_ultrasat_time_series, simulate_ultrasat_light_curve


@pytest.mark.parametrize('alpha', [0.5, 1., 1.5])
def test_bounded_sampler_preserves_one_global_uniform_draw_per_sample(alpha):
    from miletos.simulation import sample_power_law

    state = np.random.get_state()
    try:
        np.random.seed(65)
        uniform = np.random.random((2, 3))
        expected_next = np.random.random()
        np.random.seed(65)
        values = sample_power_law(0.5, 5., alpha, size=(2, 3))
        assert np.random.random() == expected_next
        if alpha == 1.:
            expected = 0.5 * 10.**uniform
        else:
            exponent = 1 - alpha
            expected = (0.5**exponent + (5.**exponent - 0.5**exponent) * uniform)**(1 / exponent)
        np.testing.assert_array_equal(values, expected)
    finally:
        np.random.set_state(state)


def test_orbit_mass_conventions_are_explicit_and_equivalent():
    from miletos.simulation import SOLAR_MASS_KG, semimajor_axis_from_mass_kg, semimajor_axis_from_period

    mass = np.array([0.6, 1., 2.])  # [M_sun]
    periods = np.array([1., 10., 365.25])  # [day]
    expected = ((6.6743e-11 * mass * SOLAR_MASS_KG * (periods * 24 * 3600)**2)
                / (4 * np.pi**2))**(1 / 3)
    np.testing.assert_array_equal(semimajor_axis_from_period(mass, periods), expected)
    np.testing.assert_array_equal(semimajor_axis_from_mass_kg(mass * SOLAR_MASS_KG, periods), expected)


def test_batman_model_retains_normalization_epoch_and_duration_placeholder():
    import setuptools
    import batman
    from miletos.simulation import generate_batman_light_curve, semimajor_axis_from_period

    time = np.linspace(3., 8., 2000)  # [day]
    parameters = batman.TransitParams()
    parameters.t0 = min(time)
    parameters.per = 2.  # [day]
    parameters.rp = 0.5
    parameters.a = semimajor_axis_from_period(0.6, 2.) / (0.01 * 6.957e8)
    parameters.inc = 90.  # [deg]
    parameters.ecc = 0
    parameters.w = 90.  # [deg]
    parameters.u = []
    parameters.fp = 0.1
    parameters.limb_dark = 'uniform'
    parameters.t_secondary = 0.5
    expected = batman.TransitModel(parameters, time, transittype='primary').light_curve(parameters)
    expected /= np.median(expected)
    actual_time, flux, duration = generate_batman_light_curve(0.01, 0.6, 0.005, 0.1, 2., 90., time)
    np.testing.assert_array_equal(actual_time, time)
    np.testing.assert_array_equal(flux, expected)
    assert duration == 0


def test_generate_ultrasat_time_series_defaults_preserve_existing_schedule():
    time_jd, cadence_seconds, exposure_seconds = generate_ultrasat_time_series()

    assert time_jd.size == 288
    assert time_jd[0] == 2462000.0
    assert np.allclose(np.diff(time_jd), 300.0 / 86400.0)
    assert cadence_seconds == 300.0
    assert exposure_seconds == 240.0


def test_generate_ultrasat_time_series_accepts_arbitrary_observation_window():
    time_jd, cadence_seconds, exposure_seconds = generate_ultrasat_time_series(
        start_jd=2463000.25,
        duration_days=0.01,
        cadence_seconds=120.0,
        exposure_seconds=90.0,
    )

    assert time_jd[0] == 2463000.25
    assert time_jd[-1] < 2463000.25 + 0.01
    assert cadence_seconds == 120.0
    assert exposure_seconds == 90.0


@pytest.mark.parametrize(
    "settings",
    [
        {"duration_days": 0.0},
        {"cadence_seconds": 0.0},
        {"cadence_seconds": np.nan},
        {"exposure_seconds": 301.0},
    ],
)
def test_generate_ultrasat_time_series_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        generate_ultrasat_time_series(**settings)


def test_simulate_ultrasat_light_curve_supports_arbitrary_models_reproducibly():
    model = lambda time: 1.0 + 0.01 * np.sin(2.0 * np.pi * (time - time[0]))
    first = simulate_ultrasat_light_curve(
        model,
        magnitude=18.0,
        start_jd=2463000.25,
        duration_days=0.02,
        seed=4,
    )
    second = simulate_ultrasat_light_curve(
        model,
        magnitude=18.0,
        start_jd=2463000.25,
        duration_days=0.02,
        seed=4,
    )

    assert first.shape == (6, 3)
    assert np.array_equal(first, second)
    assert np.all(first[:, 2] > 0.0)
    assert np.all(np.isfinite(first))


def test_simulate_ultrasat_light_curve_rejects_mismatched_model_shape():
    with pytest.raises(ValueError, match="one finite value per sample time"):
        simulate_ultrasat_light_curve(np.ones(2), magnitude=18.0, duration_days=0.02)


def test_simulate_ultrasat_light_curve_is_publicly_exported():
    assert miletos.simulate_ultrasat_light_curve is simulate_ultrasat_light_curve