import numpy as np
import pytest

from miletos.rotation_regression import run_rotation_regression_benchmark


def test_benchmark_predicts_only_held_out_periods_and_is_reproducible():
    pytest.importorskip("sklearn")
    result = run_rotation_regression_benchmark(number_samples=24, number_times=32, number_trees=4)
    repeated = run_rotation_regression_benchmark(number_samples=24, number_times=32, number_trees=4)

    assert result["predicted_period_days"].shape == result["test_indices"].shape == (5,)
    assert np.intersect1d(result["train_indices"], result["test_indices"]).size == 0
    np.testing.assert_allclose(result["predicted_period_days"], repeated["predicted_period_days"])
    np.testing.assert_allclose(
        result["relative_period_errors"],
        np.abs(result["predicted_period_days"] - result["period_days"][result["test_indices"]])
        / result["period_days"][result["test_indices"]],
    )
    assert np.isfinite(result["light_curves"]).all()


def test_benchmark_rejects_empty_test_sample():
    pytest.importorskip("sklearn")
    with pytest.raises(ValueError, match="held-out"):
        run_rotation_regression_benchmark(number_samples=2, training_fraction=1.0)