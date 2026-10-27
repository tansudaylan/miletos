from pathlib import Path
import logging

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from miletos.tess_transit_survey import (
    download_qlp_light_curves,
    injection_recovery,
    merge_qlp_light_curves,
    read_qlp_light_curve,
    search_tess_target,
)
import miletos.tess_transit_survey as tess_transit_survey
from miletos.tess_transit_search import (
    analyze_tess_target_catalog,
    estimate_survey_occurrence,
    prepare_toi_target_list,
)


@pytest.mark.parametrize('verbosity', [None, '0', '1'])
def test_qlp_download_progress_obeys_tdpy_verbosity(monkeypatch, tmp_path, capsys, verbosity):
    from astropy.table import Table

    product = tmp_path / 'hlsp_qlp_tess_ffi_s0030-0000000380907942_tess_v01_llc.fits'
    product.write_bytes(b'Retained test product')
    calls = []
    previous_level = tess_transit_survey.astroquery_log.level
    monkeypatch.setattr(tess_transit_survey.Observations, 'query_criteria',
                        lambda **kwargs: Table({'obsid': [1]}))
    monkeypatch.setattr(tess_transit_survey.Observations, 'get_product_list',
                        lambda observations: Table({'productFilename': [product.name]}))

    def download(products, **kwargs):
        expected_level = previous_level if verbosity == '1' else max(previous_level, logging.WARNING)
        assert tess_transit_survey.astroquery_log.level == expected_level
        assert tess_transit_survey.astroquery_log.isEnabledFor(logging.WARNING) == (previous_level <= logging.WARNING)
        calls.append(kwargs)
        return Table({'Local Path': [str(product)]})

    monkeypatch.setattr(tess_transit_survey.Observations, 'download_products', download)
    if verbosity is None:
        monkeypatch.delenv('TDPY_VERBOSITY', raising=False)
    else:
        monkeypatch.setenv('TDPY_VERBOSITY', verbosity)

    result = download_qlp_light_curves([380907942], [30], tmp_path)

    assert result == [product]
    assert calls == [{'download_dir': str(tmp_path), 'verbose': verbosity == '1'}]
    assert tess_transit_survey.astroquery_log.level == previous_level
    output = capsys.readouterr().out
    if verbosity == '1':
        assert f'Writing to {tmp_path}...' in output
    else:
        assert output == ''


def test_qlp_download_failure_restores_logger_and_propagates(monkeypatch, tmp_path):
    from astropy.table import Table

    previous_level = tess_transit_survey.astroquery_log.level
    monkeypatch.setenv('TDPY_VERBOSITY', '0')
    monkeypatch.setattr(tess_transit_survey.Observations, 'query_criteria',
                        lambda **kwargs: Table({'obsid': [1]}))
    monkeypatch.setattr(tess_transit_survey.Observations, 'get_product_list',
                        lambda observations: Table({'productFilename': ['example_llc.fits']}))

    def fail_download(products, **kwargs):
        raise RuntimeError('Download failed')

    monkeypatch.setattr(tess_transit_survey.Observations, 'download_products', fail_download)
    with pytest.raises(RuntimeError, match='Download failed'):
        download_qlp_light_curves([380907942], [30], tmp_path)
    assert tess_transit_survey.astroquery_log.level == previous_level


def _write_qlp_fits(path: Path, time, flux, uncertainty, quality=None, flux_name="KSPSAP_FLUX"):
    columns = [
        fits.Column(name="TIME", format="D", array=np.asarray(time)),
        fits.Column(name=flux_name, format="D", array=np.asarray(flux)),
        fits.Column(name=f"{flux_name}_ERR", format="D", array=np.asarray(uncertainty)),
    ]
    if quality is not None:
        columns.append(fits.Column(name="QUALITY", format="J", array=np.asarray(quality)))
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path)


def test_read_qlp_fits_normalizes_btjd_and_applies_quality_mask(tmp_path):
    path = tmp_path / "hlsp_qlp_tess_ffi_s0010-0000000260128333_tess_v01_llc.fits"
    _write_qlp_fits(
        path,
        [2457010.0, 2457010.02, 2457010.04, 2457010.06],  # [BJD day]
        [100.0, 99.0, 100.0, 100.0],
        [1.0, 1.0, 1.0, 1.0],
        [0, 0, 1, 0],
    )

    curve = read_qlp_light_curve(path)

    assert curve["tic_id"].unique().tolist() == [260128333]
    assert curve["sector"].unique().tolist() == [10]
    np.testing.assert_allclose(curve["time_btjd"], [10.0, 10.02, 10.06])
    assert curve["flux"].median() == 1.0


def test_read_qlp_fits_uses_matching_detrended_flux_error_pair(tmp_path):
    path = tmp_path / "hlsp_qlp_tess_ffi_s0062-0000000038455598_tess_v01_llc.fits"
    _write_qlp_fits(
        path,
        [62.0, 62.02, 62.04],
        [10.0, 9.9, 10.1],
        [0.1, 0.1, 0.1],
        flux_name="DET_FLUX",
    )

    curve = read_qlp_light_curve(path)

    assert curve["sector"].unique().tolist() == [62]
    np.testing.assert_allclose(curve["flux"], [1.0, 0.99, 1.01])
    np.testing.assert_allclose(curve["flux_error"], [0.01, 0.01, 0.01])


def test_merge_removes_duplicate_cadences_and_orders_sectors():
    first = pd.DataFrame({
        "tic_id": [10, 10], "sector": [1, 1], "time_btjd": [2.0, 1.0],  # [day]
        "flux": [1.0, 1.0], "flux_error": [0.001, 0.001],
    })
    second = pd.DataFrame({
        "tic_id": [10, 10], "sector": [2, 2], "time_btjd": [2.0, 3.0],  # [day]
        "flux": [1.0, 0.99], "flux_error": [0.001, 0.001],
    })

    merged = merge_qlp_light_curves([first, second])

    assert merged["time_btjd"].tolist() == [1.0, 2.0, 3.0]
    assert merged["sector"].tolist() == [1, 1, 2]


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
@pytest.mark.parametrize('inherit', [False, True])
def test_target_search_plot_reuses_shared_orbital_phase(tmp_path, monkeypatch, typeplotback, inherit):
    from importlib import import_module
    from unittest.mock import Mock
    import matplotlib.pyplot as plt
    from tdpy.exoplanet import orbital_phase
    from tdpy.plotting import plot_style

    search = import_module("miletos.tess_transit_search")
    shared_phase = Mock(wraps=orbital_phase)
    monkeypatch.setattr(search, "orbital_phase", shared_phase)
    time = np.array([-3.0, -0.1, 0.0, 0.1, 1.5, 3.0])  # [day]
    curve = pd.DataFrame({
        "time_btjd": time, "flux": np.ones(time.size),
        "flux_error": np.full(time.size, 0.001),
    })
    candidate = {
        "period_days": 3.0, "transit_time_days": 0.0, "duration_days": 0.2,  # [day]
        "depth": 0.01, "depth_snr": 10.0, "disposition": "candidate",
    }
    options = {} if inherit else {'typeplotback': typeplotback}
    outer_theme = typeplotback if inherit else ('dark' if typeplotback == 'white' else 'white')
    with plot_style(outer_theme):
        original = dict(plt.rcParams)
        path = search.plot_target_search(curve, candidate, tmp_path / "target.png", **options)
        assert dict(plt.rcParams) == original
    shared_phase.assert_called_once()
    arguments, keywords = shared_phase.call_args
    np.testing.assert_array_equal(arguments[0], time)
    assert arguments[1:] == (3.0, 0.0)
    assert keywords == {"center": 0.0}
    pixels = plt.imread(path)
    assert pixels.shape[0] > 100 and pixels.shape[1] > 100
    assert np.ptp(pixels[..., :3]) > 0.5
    np.testing.assert_array_equal(pixels[0, 0], (0, 0, 0, 1) if typeplotback == 'dark' else (1, 1, 1, 1))


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
@pytest.mark.parametrize('inherit', [False, True])
def test_survey_products_theme_all_outputs_and_preserve_tables(tmp_path, monkeypatch, typeplotback, inherit):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from PIL import Image
    from miletos.tess_transit_search import plot_survey_products
    from tdpy.plotting import plot_background_colors, plot_style

    results = pd.DataFrame({
        'searched': [True, True, False], 'period_days': [3.0, 4.0, np.nan],
        'depth_snr': [12.0, 4.0, np.nan], 'inverted_depth_snr': [2.0, 1.0, np.nan],
        'detection': [True, False, False], 'tmag': [11.0, 12.0, 13.0],
        'reference_disposition': ['CP', 'FP', 'PC'], 'disposition': ['candidate', 'review', None],
    })
    injections = pd.DataFrame({'period_days': [3.0, 4.0], 'depth': [0.01, 0.02], 'completeness': [0.7, 0.9]})
    original_results, original_injections = results.copy(deep=True), injections.copy(deep=True)
    background, foreground = plot_background_colors(typeplotback)
    save = plt.Figure.savefig

    def capture_and_save(figure, path, **kwargs):
        assert figure.get_facecolor() == to_rgba(background)
        assert all(axis.xaxis.label.get_color() == foreground for axis in figure.axes)
        return save(figure, path, **kwargs)

    monkeypatch.setattr(plt.Figure, 'savefig', capture_and_save)
    options = {} if inherit else {'typeplotback': typeplotback}
    outer_theme = typeplotback if inherit else ('dark' if typeplotback == 'white' else 'white')
    with plot_style(outer_theme):
        original_settings = dict(plt.rcParams)
        paths = plot_survey_products(results, injections, tmp_path, **options)
        assert dict(plt.rcParams) == original_settings
    assert set(paths) == {'search_yield', 'completeness', 'false_alarm_control', 'magnitude_yield', 'toi_comparison'}
    for path in paths.values():
        with Image.open(path) as image:
            expected = tuple(int(channel * 255) for channel in to_rgba(background)[:3])
            assert image.convert('RGB').getpixel((0, 0)) == expected
    pd.testing.assert_frame_equal(results, original_results)
    pd.testing.assert_frame_equal(injections, original_injections)


def test_tess_search_reports_flags_and_inverted_control():
    time = np.arange(0.0, 30.0, 0.02)  # [day]
    period = 3.0  # [day]
    phase = (time - 0.4 + period / 2.0) % period - period / 2.0
    flux = np.ones_like(time)
    flux[np.abs(phase) < 0.06] -= 0.01
    flux += np.random.default_rng(9).normal(0.0, 0.001, time.size)
    curve = pd.DataFrame({
        "time_btjd": time, "flux": flux,
        "flux_error": np.full(time.size, 0.001),
    })

    result = search_tess_target(
        curve, minimum_period_days=2.5, maximum_period_days=3.5,  # [day]
        duration_days=[0.10, 0.12, 0.14],  # [day]
    )

    assert result["period_days"] == pytest.approx(period, abs=0.02)
    assert result["depth_snr"] > 20
    assert "inverted_depth_snr" in result
    assert result["disposition"] in {"candidate", "review"}


def test_injection_recovery_is_reproducible_and_uses_observed_noise():
    time = np.arange(0.0, 30.0, 0.02)  # [day]
    curve = pd.DataFrame({
        "time_btjd": time,
        "flux": 1.0 + np.random.default_rng(4).normal(0.0, 0.001, time.size),
        "flux_error": np.full(time.size, 0.001),
    })
    first = injection_recovery(curve, [3.0], [0.01], 0.10, trials=3, seed=12)  # [day]
    second = injection_recovery(curve, [3.0], [0.01], 0.10, trials=3, seed=12)  # [day]

    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "recoveries"] >= 2


def test_injection_recovery_requires_the_full_vetter_to_pass(monkeypatch):
    time = np.arange(0.0, 30.0, 0.1)  # [day]
    curve = pd.DataFrame({
        "time_btjd": time,
        "flux": np.ones_like(time),
        "flux_error": np.full(time.size, 0.001),
    })
    search_intervals = []

    def fake_search(time_days, flux, uncertainty, *, duration_days,
                    minimum_period_days, maximum_period_days):
        search_intervals.append((minimum_period_days, maximum_period_days))
        return {
            "period_days": 3.0,
            "transit_time_days": 0.0,
            "duration_days": 0.1,
            "depth": 0.01,
            "depth_uncertainty": 0.001,
            "depth_snr": 10.0,
        }

    def flag_for_review(light_curve, candidate, *, snr_threshold, check_inverted_control):
        assert not check_inverted_control
        return {**candidate, "disposition": "review"}

    monkeypatch.setattr(tess_transit_survey, "search_box_least_squares", fake_search)
    monkeypatch.setattr(tess_transit_survey, "vet_transit_candidate", flag_for_review)

    recovery = injection_recovery(curve, [3.0], [0.01], 0.1, trials=1, seed=2)

    assert recovery.loc[0, "recoveries"] == 0
    assert recovery.loc[0, "period_matched"] == 1
    assert recovery.loc[0, "snr_matched"] == 1
    assert recovery.loc[0, "passed_vetter"] == 0
    assert search_intervals == [(0.5, pytest.approx(29.9 / 3.0))]


def test_prepare_toi_list_applies_sector_and_qlp_magnitude_limits(tmp_path):
    path = tmp_path / "toi.csv"
    pd.DataFrame({
        "TIC ID": [1, 2, 3], "TOI": [101.01, 102.01, 103.01],
        "TESS Mag": [11.0, 12.0, 13.8], "Sectors": ["10,11", "11", "10"],
        "TESS Disposition": ["PC", "KP", "PC"],
    }).to_csv(path, index=False)

    selected = prepare_toi_target_list(path, sector=10)

    assert selected[["tic_id", "sector", "tmag"]].to_dict("records") == [
        {"tic_id": 1, "sector": 10, "tmag": 11.0}
    ]
    assert selected.loc[0, "reference_disposition"] == "PC"
    assert not selected.loc[0, "sample_is_parent_population"]


def test_offline_survey_retains_targets_without_qlp_products(tmp_path):
    targets = pd.DataFrame({"tic_id": [260128333], "sector": [10], "tmag": [11.5]})

    results, injections = analyze_tess_target_catalog(
        targets, tmp_path / "visuals", download=False, injection_trials=0,
        data_directory=tmp_path / "empty_data",
    )

    assert len(results) == 1
    assert not bool(results.loc[0, "searched"])
    assert results.loc[0, "failure"] == "no_qlp_product"
    assert injections.empty


@pytest.fixture
def occurrence_tables():
    results = pd.DataFrame({
        "tic_id": [1, 2], "searched": [True, True], "detection": [True, False],
        "sample_is_parent_population": [True, True],
        "inverted_control_disposition": ["review", "review"],
    })
    injections = pd.DataFrame({
        "tic_id": [1, 1, 2, 2], "period_days": [3.0, 10.0] * 2,  # [day]
        "depth": [0.001, 0.001] * 2, "completeness": [0.8, 0.8, 0.5, 0.5],
    })
    return results, injections


def test_occurrence_requires_complete_parent_sample_and_uses_pergamon(occurrence_tables):
    results, injections = occurrence_tables

    with pytest.raises(ValueError, match="complete parent"):
        estimate_survey_occurrence(results.drop(columns="sample_is_parent_population"), injections)
    summary = estimate_survey_occurrence(results, injections)

    assert summary["searched_targets"] == 2
    assert summary["detections"] == 1
    assert 0.0 <= summary["uniform_prior_posterior_median"] <= 1.0
    assert len(summary["uniform_prior_68_percent_interval"]) == 2


@pytest.mark.parametrize("column", ["sample_is_parent_population", "searched", "detection"])
@pytest.mark.parametrize("invalid", ["False", np.nan, 0.9, 2, 1j])
def test_occurrence_rejects_invalid_flags_before_pergamon(monkeypatch, occurrence_tables, column, invalid):
    from unittest.mock import Mock
    import pergamon

    results, injections = occurrence_tables
    results[column] = [invalid, True]
    original_results = results.copy(deep=True)
    original_injections = injections.copy(deep=True)
    estimator = Mock(side_effect=AssertionError("Invalid flags reached occurrence inference"))
    monkeypatch.setattr(pergamon, "estimate_occurrence_rate", estimator)

    with pytest.raises(ValueError, match=f"{column}.*Boolean or binary"):
        estimate_survey_occurrence(results, injections)

    estimator.assert_not_called()
    pd.testing.assert_frame_equal(results, original_results)
    pd.testing.assert_frame_equal(injections, original_injections)


@pytest.mark.parametrize("dtype", [bool, int, float])
def test_occurrence_preserves_valid_binary_flags_and_inputs(occurrence_tables, dtype):
    results, injections = occurrence_tables
    expected = estimate_survey_occurrence(results, injections)
    for column in ("sample_is_parent_population", "searched", "detection"):
        results[column] = results[column].astype(dtype)
    original_results = results.copy(deep=True)
    original_injections = injections.copy(deep=True)

    actual = estimate_survey_occurrence(results, injections)

    assert actual == expected
    pd.testing.assert_frame_equal(results, original_results)
    pd.testing.assert_frame_equal(injections, original_injections)


@pytest.mark.parametrize("kind", [
    "missing_completeness", "infinite_completeness", "negative_completeness",
    "above_one_completeness", "complex_completeness", "duplicate_target",
    "duplicate_injection", "unmatched_injection_target", "missing_injection_target",
])
def test_occurrence_rejects_incomplete_or_duplicate_bookkeeping(monkeypatch, occurrence_tables, kind):
    from unittest.mock import Mock
    import pergamon

    results, injections = occurrence_tables
    if kind.endswith("completeness"):
        invalid = {
            "missing_completeness": np.nan, "infinite_completeness": np.inf,
            "negative_completeness": -0.1, "above_one_completeness": 1.1,
            "complex_completeness": 0.8 + 1j,
        }[kind]
        if kind == "complex_completeness":
            injections["completeness"] = injections["completeness"].astype(complex)
        injections.loc[1, "completeness"] = invalid
    elif kind == "duplicate_target":
        results = pd.concat([results, results.iloc[[0]]], ignore_index=True)
    elif kind == "duplicate_injection":
        injections = pd.concat([injections, injections.iloc[[2]]], ignore_index=True)
    elif kind == "unmatched_injection_target":
        injections.loc[injections["tic_id"] == 2, "tic_id"] = 3
    else:
        injections = injections[injections["tic_id"] == 1].copy()
    original_results = results.copy(deep=True)
    original_injections = injections.copy(deep=True)
    estimator = Mock(side_effect=AssertionError("Invalid survey records reached occurrence inference"))
    monkeypatch.setattr(pergamon, "estimate_occurrence_rate", estimator)

    with pytest.raises(ValueError):
        estimate_survey_occurrence(results, injections)

    estimator.assert_not_called()
    pd.testing.assert_frame_equal(results, original_results)
    pd.testing.assert_frame_equal(injections, original_injections)


@pytest.mark.parametrize("invalid", [None, np.nan, "", "Candidate", "unvetted"])
def test_occurrence_requires_valid_native_inverted_control_labels(monkeypatch, occurrence_tables, invalid):
    from unittest.mock import Mock
    import pergamon

    results, injections = occurrence_tables
    results["inverted_control_disposition"] = [invalid, "review"]
    estimator = Mock(side_effect=AssertionError("Unknown inverted controls reached inference"))
    monkeypatch.setattr(pergamon, "estimate_occurrence_rate", estimator)

    with pytest.raises(ValueError, match="inverted_control_disposition"):
        estimate_survey_occurrence(results, injections)

    estimator.assert_not_called()