from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from miletos.tess_transit_survey import (
    injection_recovery,
    merge_qlp_light_curves,
    read_qlp_light_curve,
    search_tess_target,
)
from miletos.tess_transit_search import (
    analyze_tess_target_catalog,
    estimate_survey_occurrence,
    prepare_toi_target_list,
)


def _write_qlp_fits(path: Path, time, flux, uncertainty, quality=None):
    columns = [
        fits.Column(name="TIME", format="D", array=np.asarray(time)),
        fits.Column(name="KSPSAP_FLUX", format="D", array=np.asarray(flux)),
        fits.Column(name="KSPSAP_FLUX_ERR", format="D", array=np.asarray(uncertainty)),
    ]
    if quality is not None:
        columns.append(fits.Column(name="QUALITY", format="J", array=np.asarray(quality)))
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path)


def test_read_qlp_fits_normalizes_btjd_and_applies_quality_mask(tmp_path):
    path = tmp_path / "hlsp_qlp_tess_ffi_s0010-0000000260128333_tess_v01_llc.fits"
    _write_qlp_fits(
        path,
        [2457010.0, 2457010.02, 2457010.04, 2457010.06],
        [100.0, 99.0, 100.0, 100.0],
        [1.0, 1.0, 1.0, 1.0],
        [0, 0, 1, 0],
    )

    curve = read_qlp_light_curve(path)

    assert curve["tic_id"].unique().tolist() == [260128333]
    assert curve["sector"].unique().tolist() == [10]
    np.testing.assert_allclose(curve["time_btjd"], [10.0, 10.02, 10.06])
    assert curve["flux"].median() == 1.0


def test_merge_removes_duplicate_cadences_and_orders_sectors():
    first = pd.DataFrame({
        "tic_id": [10, 10], "sector": [1, 1], "time_btjd": [2.0, 1.0],
        "flux": [1.0, 1.0], "flux_error": [0.001, 0.001],
    })
    second = pd.DataFrame({
        "tic_id": [10, 10], "sector": [2, 2], "time_btjd": [2.0, 3.0],
        "flux": [1.0, 0.99], "flux_error": [0.001, 0.001],
    })

    merged = merge_qlp_light_curves([first, second])

    assert merged["time_btjd"].tolist() == [1.0, 2.0, 3.0]
    assert merged["sector"].tolist() == [1, 1, 2]


def test_tess_search_reports_flags_and_inverted_control():
    time = np.arange(0.0, 30.0, 0.02)
    period = 3.0
    phase = (time - 0.4 + period / 2.0) % period - period / 2.0
    flux = np.ones_like(time)
    flux[np.abs(phase) < 0.06] -= 0.01
    flux += np.random.default_rng(9).normal(0.0, 0.001, time.size)
    curve = pd.DataFrame({
        "time_btjd": time, "flux": flux,
        "flux_error": np.full(time.size, 0.001),
    })

    result = search_tess_target(
        curve, minimum_period_days=2.5, maximum_period_days=3.5,
        duration_days=[0.10, 0.12, 0.14],
    )

    assert result["period_days"] == pytest.approx(period, abs=0.02)
    assert result["depth_snr"] > 20
    assert "inverted_depth_snr" in result
    assert result["disposition"] in {"candidate", "review"}


def test_injection_recovery_is_reproducible_and_uses_observed_noise():
    time = np.arange(0.0, 30.0, 0.02)
    curve = pd.DataFrame({
        "time_btjd": time,
        "flux": 1.0 + np.random.default_rng(4).normal(0.0, 0.001, time.size),
        "flux_error": np.full(time.size, 0.001),
    })
    first = injection_recovery(curve, [3.0], [0.01], 0.10, trials=3, seed=12)
    second = injection_recovery(curve, [3.0], [0.01], 0.10, trials=3, seed=12)

    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "recoveries"] >= 2


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


def test_occurrence_requires_complete_parent_sample_and_uses_pergamon():
    results = pd.DataFrame({
        "tic_id": [1, 2], "searched": [True, True], "detection": [True, False],
        "sample_is_parent_population": [True, True],
    })
    injections = pd.DataFrame({
        "tic_id": [1, 1, 2, 2], "period_days": [3.0, 10.0] * 2,
        "depth": [0.001, 0.001] * 2, "completeness": [0.8, 0.8, 0.5, 0.5],
    })

    with pytest.raises(ValueError, match="complete parent"):
        estimate_survey_occurrence(results.drop(columns="sample_is_parent_population"), injections)
    summary = estimate_survey_occurrence(results, injections)

    assert summary["searched_targets"] == 2
    assert summary["detections"] == 1
    assert 0.0 <= summary["uniform_prior_posterior_median"] <= 1.0
    assert len(summary["uniform_prior_68_percent_interval"]) == 2