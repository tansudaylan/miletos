import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import EarthLocation

from miletos.rubin_transit_search import (
    prepare_rubin_light_curve,
    query_rubin_forced_photometry,
    search_rubin_target,
)


class _Job:
    phase = "COMPLETED"

    def __init__(self, result):
        self.result = result
        self.deleted = False

    def run(self):
        pass

    def wait(self, phases):
        assert phases == ["COMPLETED", "ERROR"]

    def fetch_result(self):
        return self.result

    def delete(self):
        self.deleted = True


class _Service:
    def __init__(self, result, visit_columns=None):
        self.result = result
        self.queries = []
        self.job = _Job(result)
        self.visit_columns = visit_columns or [
            "visitId", "expMidptMJD", "obsStartMJD", "expTime", "band"
        ]

    def search(self, query):
        self.queries.append(query)
        if "ForcedSource" in query:
            columns = ["objectId", "visit", "psfFlux", "psfFluxErr", "psfFlux_flag"]
        else:
            columns = self.visit_columns
        return pd.DataFrame({"column_name": columns})

    def submit_job(self, query):
        self.queries.append(query)
        return self.job


def test_query_uses_schema_discovery_indexed_object_id_and_async_job():
    rows = pd.DataFrame({"object_id": [42], "visit_id": [7], "band": ["r"]})
    service = _Service(rows)

    result = query_rubin_forced_photometry(service, "dp1", 42)

    assert result.equals(rows)
    assert service.job.deleted
    assert "fs.objectId = 42" in service.queries[-1]
    assert "JOIN dp1.Visit" in service.queries[-1]
    assert "v.expMidptMJD AS time_mjd" in service.queries[-1]
    assert "v.expTime AS exposure_time_s" in service.queries[-1]
    with pytest.raises(ValueError, match="positive integer"):
        query_rubin_forced_photometry(service, "dp1", -1)


def test_query_builds_midpoint_from_visit_start_and_exposure_when_needed():
    service = _Service(
        pd.DataFrame(),
        visit_columns=["visitId", "observationStartMJD", "exposureTime", "band"],
    )

    query_rubin_forced_photometry(service, "edp2", 42)

    assert "(v.observationStartMJD + v.exposureTime / 172800.0) AS time_mjd" in service.queries[-1]


def test_prepare_converts_tai_midpoints_and_normalizes_each_band():
    rows = pd.DataFrame({
        "object_id": [1, 1, 1, 1, 1], "visit_id": [1, 2, 3, 4, 5],
        "band": ["g", "g", "r", "r", "r"],
        "time_mjd": [61000.0, 61001.0, 61000.5, 61001.5, 61002.5],
        "psf_flux_njy": [100.0, 90.0, 200.0, 180.0, 999.0],
        "psf_flux_error_njy": [2.0, 2.0, 4.0, 4.0, 4.0],
        "psf_flux_flag": [False, False, False, False, True],
    })
    site = EarthLocation.from_geodetic(lon=-70.7494, lat=-30.2444, height=2663.0)

    light_curve = prepare_rubin_light_curve(
        rows, ra_deg=53.0, dec_deg=-28.0,
        observatory_location=site, time_scale="tai",
    )

    assert len(light_curve) == 4
    assert set(light_curve["band"]) == {"g", "r"}
    np.testing.assert_allclose(
        light_curve["flux"], [100.0 / 95.0, 200.0 / 190.0, 90.0 / 95.0, 180.0 / 190.0]
    )
    assert np.all(np.diff(light_curve["time_bjd_tdb"]) > 0.0)


def test_search_recovers_synthetic_transits_without_claiming_vetting():
    time = np.arange(0.0, 30.0, 0.05)
    period = 3.0
    phase = (time - 0.4 + 0.5 * period) % period - 0.5 * period
    flux = np.ones_like(time)
    flux[np.abs(phase) < 0.06] -= 0.01
    flux += np.random.default_rng(8).normal(0.0, 0.001, time.size)
    light_curve = pd.DataFrame({
        "time_bjd_tdb": time,
        "flux": flux,
        "flux_error": np.full(time.size, 0.001),
        "band": "r",
    })

    candidates = search_rubin_target(
        light_curve,
        minimum_period_days=2.5,
        maximum_period_days=3.5,
        duration_days=[0.10, 0.12, 0.14],
    )

    assert candidates
    assert candidates[0]["period_days"] == pytest.approx(period, abs=0.02)
    assert candidates[0]["transit_count"] >= 3
    assert candidates[0]["disposition"] == "unvetted"