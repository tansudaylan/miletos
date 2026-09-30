import numpy as np

from miletos.daylan2021 import (
    PAPER_DOI,
    PUBLISHED_TRANSITS,
    RecoveredTransit,
    _sigma_differences,
    phase_time,
    trapezoid_transit,
)


def test_daylan2021_publication_contract_is_four_planets():
    assert PAPER_DOI == '10.3847/1538-3881/abd73e'
    assert [transit.planet for transit in PUBLISHED_TRANSITS] == list('bcde')
    assert np.allclose(
        [transit.period_days for transit in PUBLISHED_TRANSITS],
        [3.79523, 6.20370, 14.17555, 19.5917],
    )


def test_periodic_trapezoid_has_expected_depth_and_baseline():
    time = np.array([0.0, 0.05, 0.1, 0.5, 2.0])
    model = trapezoid_transit(time, 0.0, 2.0, 0.01, 0.2, 0.2)

    assert np.allclose(model[[0, 4]], 0.99)
    assert model[1] == 0.99
    assert model[2] == 1.0
    assert model[3] == 1.0
    assert np.allclose(phase_time(time, 0.0, 2.0), [0.0, 0.05, 0.1, 0.5, 0.0])


def test_identical_recovery_has_zero_sigma_difference():
    recovered = tuple(
        RecoveredTransit(
            transit.planet,
            transit.epoch_bjd,
            transit.epoch_uncertainty_days,
            transit.period_days,
            transit.period_uncertainty_days,
            transit.depth,
            transit.depth_uncertainty,
            transit.duration_days,
            transit.duration_uncertainty_days,
            0.15,
        )
        for transit in PUBLISHED_TRANSITS
    )

    assert _sigma_differences(PUBLISHED_TRANSITS, recovered) == [0.0] * 16