import numpy as np
import pytest

import miletos
from miletos.simulation import generate_ultrasat_time_series, simulate_ultrasat_light_curve


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