import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "TIC266680498"
    / "run.py"
)
SPEC = importlib.util.spec_from_file_location("miletos_tic266680498_example", EXAMPLE_PATH)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


def test_cached_pdcsap_bjd_is_converted_to_btjd_without_dropping_rows(tmp_path):
    path = tmp_path / "sector.csv"
    print(f"Writing to {path}...")
    path.write_text(
        "# time, relative flux, relative flux error\n"
        "2457001.0,1.001,0.001\n"
        "2457001.1,0.999,0.001\n"
        "2457001.2,1.000,0.002\n",
        encoding="utf-8",
    )

    curve = EXAMPLE.read_cached_sector_light_curve(path, 3)

    np.testing.assert_allclose(curve["time_btjd"], [1.0, 1.1, 1.2], atol=3e-10)
    np.testing.assert_array_equal(curve["flux"], [1.001, 0.999, 1.0])
    assert curve["sector"].tolist() == [3, 3, 3]


@pytest.mark.parametrize("rows", [
    "2457001,1,0.001,4\n2457002,2,0.001,4\n2457003,3,0.001,4\n",
    "2457001,1,0\n2457002,1,0.001\n2457003,1,0.001\n",
    "2457001,1,0.001\n2457002,nan,0.001\n2457003,1,0.001\n",
    "2457001,1,0.001\n2457001,1,0.001\n2457003,1,0.001\n",
])
def test_malformed_cached_photometry_is_rejected(tmp_path, rows):
    path = tmp_path / "sector.csv"
    path.write_text(rows, encoding="utf-8")
    with pytest.raises(ValueError):
        EXAMPLE.read_cached_sector_light_curve(path, 3)


@pytest.mark.parametrize("include_combined", [False, True])
@pytest.mark.parametrize("mutate_source", [False, True])
def test_run_analysis_uses_miletos_search_for_each_sector_and_optional_combined_curve(
    tmp_path, monkeypatch, include_combined, mutate_source
):
    data_directory = tmp_path / "data"
    output_directory = tmp_path / "visuals"
    data_directory.mkdir()
    searched = []
    plotted = []

    for sector in EXAMPLE.SECTORS:
        path = data_directory / f"TESS_PDCSAP_FLUX_Sector{sector:02d}.csv"
        path.write_text(
            "# time, relative flux, relative flux error\n"
            "2457001.0,1.0,0.001\n"
            "2457002.0,0.99,0.001\n"
            "2457003.0,1.0,0.001\n"
            "2457004.0,1.0,0.001\n",
            encoding="utf-8",
        )

    def fake_search(light_curve, **kwargs):
        searched.append((light_curve.copy(), kwargs))
        if mutate_source and len(searched) == 1:
            source = data_directory / f"TESS_PDCSAP_FLUX_Sector{EXAMPLE.SECTORS[0]:02d}.csv"
            source.write_text(source.read_text() + "\n", encoding="utf-8")
        return {
            "period_days": 2.0,
            "transit_time_days": 1.1,
            "duration_days": 0.1,
            "depth": 0.01,
            "depth_uncertainty": 0.001,
            "depth_snr": 10.0,
            "odd_even_snr": 0.0,
            "secondary_depth": 0.0,
            "secondary_snr": 0.0,
            "inverted_depth_snr": 0.0,
            "disposition": "review",
        }

    monkeypatch.setattr(EXAMPLE, "search_tess_target", fake_search)
    monkeypatch.setattr(
        EXAMPLE,
        "plot_target_search",
        lambda light_curve, result, path, **kwargs: plotted.append((Path(path), kwargs)),
    )

    if mutate_source:
        with pytest.raises(RuntimeError, match="changed during analysis"):
            EXAMPLE.run_analysis(data_directory=data_directory, output_directory=output_directory,
                                 include_combined=include_combined, time_tag="20261025_220000")
        assert not list(data_directory.parent.glob("*miletos_bls_results*"))
        return

    results = EXAMPLE.run_analysis(
        data_directory=data_directory,
        output_directory=output_directory,
        include_combined=include_combined,
        time_tag="20261025_220000",
    )

    expected_count = len(EXAMPLE.SECTORS) + int(include_combined)
    assert len(searched) == expected_count
    assert len(results) == expected_count
    assert len(plotted) == expected_count
    assert all(call[1]["data_label"] == EXAMPLE.DATA_LABEL for call in plotted)
    assert searched[0][1]["minimum_period_days"] == EXAMPLE.MINIMUM_PERIOD_DAYS
    expected_plot_names = [
        f"tic266680498_sector{sector:02d}_bls_search_20261025_220000.png"
        for sector in EXAMPLE.SECTORS
    ]
    if include_combined:
        expected_plot_names.append("tic266680498_combined_bls_search_20261025_220000.png")
    assert sorted(path.name for path, _ in plotted) == sorted(expected_plot_names)
    result_path = data_directory.parent / "tic266680498_miletos_bls_results_20261025_220000.csv"
    assert result_path.is_file()
    saved = pd.read_csv(result_path)
    assert len(saved) == expected_count
    manifest = json.loads(result_path.with_suffix(".json").read_text())
    assert len(manifest["input_sha256"]) == len(EXAMPLE.SECTORS)
    assert manifest["include_combined"] is include_combined
    original = result_path.read_bytes()
    with pytest.raises(FileExistsError):
        EXAMPLE.run_analysis(data_directory=data_directory, output_directory=output_directory,
                             include_combined=include_combined, time_tag="20261025_220000")
    assert result_path.read_bytes() == original
    assert len(searched) == expected_count


def test_target_search_plot_labels_cached_pdcsap_as_its_source(tmp_path, monkeypatch):
    from miletos.tess_transit_search import plot_target_search

    captured_axes = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        captured_axes.extend(axes)
        return figure, axes

    monkeypatch.setattr(plt, "subplots", capture_subplots)
    light_curve = pd.DataFrame({
        "time_btjd": [1.0, 2.0, 3.0, 4.0],
        "flux": [1.0, 0.99, 1.0, 1.0],
        "flux_error": [0.001] * 4,
    })
    result = {
        "period_days": 2.0,
        "transit_time_days": 1.0,
        "duration_days": 0.1,
        "depth": 0.01,
        "depth_snr": 10.0,
        "disposition": "review",
    }

    plot_target_search(
        light_curve,
        result,
        tmp_path / "pdcsap.png",
        data_label=EXAMPLE.DATA_LABEL,
    )

    assert captured_axes[0].get_title() == f"Observed {EXAMPLE.DATA_LABEL} light curve"