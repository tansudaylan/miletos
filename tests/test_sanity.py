import inspect

import numpy as np
import pytest

from miletos.main import init
from miletos.sanity import (
    check_model_flux_sanity,
    check_phase_fold_sanity,
    check_time_series_sanity,
    check_transit_parameter_sanity,
)


def test_pipeline_sanity_checks_are_opt_in():
    assert inspect.signature(init).parameters["boolsanity"].default is False


def test_time_series_sanity_accepts_finite_sorted_data():
    light_curve = np.array([
        [[1.0, 1.0, 0.01], [1.0, 1.0, 0.02]],
        [[2.0, 0.99, 0.01], [2.0, 1.01, 0.02]],
    ])

    check_time_series_sanity(light_curve, boolsanity=True, label="test")


@pytest.mark.parametrize(
    ("light_curve", "message"),
    [
        (np.ones((3, 2)), "shape"),
        (np.array([[[1.0, 1.0, -0.01]]]), "negative flux uncertainties"),
        (np.array([[[1.0, np.nan, 0.01]]]), "nonfinite"),
        (np.array([[[2.0, 1.0, 0.01]], [[1.0, 1.0, 0.01]]]), "sorted"),
    ],
)
def test_time_series_sanity_rejects_invalid_values(light_curve, message):
    with pytest.raises(ValueError, match=message):
        check_time_series_sanity(light_curve, boolsanity=True, label="test")


def test_time_series_sanity_is_skipped_when_disabled():
    check_time_series_sanity(np.ones((2, 2)), boolsanity=False, label="test")


def test_transit_parameter_sanity_accepts_bound_orbits_and_missing_duration():
    check_transit_parameter_sanity(
        [3.0, 10.0],  # [day]
        [2457000.0, 2457001.0],  # [BJD]
        [0.1, 0.05],
        [0.0, 0.2],
        [2.0, np.nan],  # [hour]
        [0.0, 0.1],
        [0.0, 0.1],
        boolsanity=True,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"period_days": [0.0]}, "periods must be positive"),
        ({"cosine_inclination": [1.01]}, "cosine inclinations"),
        ({"duration_hours": [24.0]}, "shorter than the orbital period"),
        ({"eccentricity_cosine": [1.0]}, "eccentricity below one"),
        ({"epoch_days": [np.inf]}, "epoch values must be finite"),
    ],
)
def test_transit_parameter_sanity_rejects_nonphysical_values(overrides, message):
    parameters = {
        "period_days": [1.0],
        "epoch_days": [2457000.0],  # [BJD]
        "scaled_semimajor_axis": [0.1],
        "cosine_inclination": [0.0],
        "duration_hours": [2.0],  # [hour]
        "eccentricity_cosine": [0.0],
        "eccentricity_sine": [0.0],
    }
    parameters.update(overrides)

    with pytest.raises(ValueError, match=message):
        check_transit_parameter_sanity(**parameters, boolsanity=True)


def test_transit_parameter_sanity_is_skipped_when_disabled():
    check_transit_parameter_sanity(
        [0.0], [np.nan], [0.0], [2.0], [-1.0], [2.0], [0.0], boolsanity=False
    )


def test_model_flux_sanity_accepts_positive_time_aligned_models():
    check_model_flux_sanity(
        [[1.0, 1.01], [0.99, 1.02]],
        [1.0, 2.0],
        boolsanity=True,
    )


@pytest.mark.parametrize(
    ("model_flux", "time_days", "message"),
    [
        ([1.0, 1.0], [1.0], "one model value per observation"),
        ([1.0, 0.0], [1.0, 2.0], "must remain positive"),
        ([1.0, np.nan], [1.0, 2.0], "must be finite"),
    ],
)
def test_model_flux_sanity_rejects_invalid_models(model_flux, time_days, message):
    with pytest.raises(ValueError, match=message):
        check_model_flux_sanity(model_flux, time_days, boolsanity=True)


def test_model_flux_sanity_is_skipped_when_disabled():
    check_model_flux_sanity([-1.0, np.nan], [np.nan], boolsanity=False)


@pytest.mark.parametrize(
    ("phase_center", "phase"),
    [(0.0, [-0.5, 0.0, 0.49]), (0.5, [0.0, 0.5, 0.99]), (0.25, [-0.25, 0.25, 0.74])],
)
def test_phase_fold_sanity_accepts_expected_cycle_ranges(phase_center, phase):
    series = np.zeros((len(phase), 1, 3))
    series[:, 0, 0] = phase

    check_phase_fold_sanity(series, phase_center=phase_center, boolsanity=True)


def test_phase_fold_sanity_rejects_unsorted_or_out_of_range_phases():
    unsorted = np.zeros((2, 1, 3))
    unsorted[:, 0, 0] = [0.2, -0.2]
    with pytest.raises(ValueError, match="must be sorted"):
        check_phase_fold_sanity(unsorted, phase_center=0.0, boolsanity=True)

    out_of_range = np.zeros((1, 1, 3))
    out_of_range[0, 0, 0] = 0.6
    with pytest.raises(ValueError, match=r"\[-0.5, 0.5\)"):
        check_phase_fold_sanity(out_of_range, phase_center=0.0, boolsanity=True)


def test_phase_fold_sanity_is_skipped_when_disabled():
    check_phase_fold_sanity(np.ones((2, 2)), phase_center=np.nan, boolsanity=False)