import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "bls_performance"
    / "run.py"
)
SPEC = importlib.util.spec_from_file_location("miletos_bls_performance_example", EXAMPLE_PATH)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


def test_synthetic_light_curve_uses_documented_box_transit():
    first = EXAMPLE.make_synthetic_light_curve(1000, seed=4)
    second = EXAMPLE.make_synthetic_light_curve(1000, seed=4)
    for first_values, second_values in zip(first, second):
        np.testing.assert_array_equal(first_values, second_values)
    time_days, flux, uncertainty = first
    np.testing.assert_array_equal(time_days[[0, -1]], [0.0, EXAMPLE.BASELINE_DAYS])
    assert np.all(uncertainty == 0.001)
    assert np.min(flux) < np.median(flux) - 0.005


def test_benchmark_uses_miletos_search_and_writes_table_and_figure(tmp_path, monkeypatch):
    calls = []
    clock_values = iter((1.0, 1.25, 2.0, 2.5, 3.0, 3.25, 4.0, 4.5))

    def fake_search(time_days, flux, flux_uncertainty, **kwargs):
        calls.append((len(time_days), kwargs["frequency_factor"]))
        return {
            "period_days": 3.2,
            "depth_snr": 12.0,
        }

    monkeypatch.setattr(EXAMPLE, "search_box_least_squares", fake_search)
    monkeypatch.setattr(EXAMPLE, "perf_counter", lambda: next(clock_values))
    results = EXAMPLE.run_benchmark(
        cadence_counts=(100, 200),
        frequency_factors=(1.0, 4.0),
        output_directory=tmp_path,
    )

    assert calls == [(100, 1.0), (100, 4.0), (200, 1.0), (200, 4.0)]
    assert results["elapsed_seconds"].tolist() == [0.25, 0.5, 0.25, 0.5]
    assert results["recovered_period_days"].tolist() == [3.2] * 4
    table_path = tmp_path / "bls_runtime_benchmark.csv"
    figure_path = tmp_path / "bls_runtime_benchmark.png"
    print(f"Reading from {table_path}...")
    saved = pd.read_csv(table_path)
    assert len(saved) == 4
    with Image.open(figure_path) as image:
        assert image.width >= 1000 and image.height >= 300
        assert any(minimum < maximum for minimum, maximum in image.getextrema())


def test_benchmark_rejects_invalid_search_grids(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="cadence_counts"):
        EXAMPLE.run_benchmark(cadence_counts=(2,), output_directory=tmp_path)
    with pytest.raises(ValueError, match="frequency_factors"):
        EXAMPLE.run_benchmark(frequency_factors=(0.0,), output_directory=tmp_path)