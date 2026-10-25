import sys
from types import ModuleType

import numpy as np
import pytest

import miletos


def _dace_data():
    return {
        "file_key": np.array(["visit-a"] * 4 + ["visit-b"] * 4),
        "obj_date_bjd_vect": np.array([59001.0, 59001.01, 59001.02, 59001.03,
                        59000.0, 59000.01, 59000.02, 59000.03]),
        "photom_flux_vect": np.array([10.0, 11.0, 10.0, 10.0, 20.0, 21.0, 19.0, 20.0]),
        "photom_flux_vect_err": np.ones(8),
        "photom_event_saa_flag": np.array([0, 0, 0, 1, 0, 0, 0, 0]),
        "photom_event_temp_flag": np.zeros(8),
        "photom_event_earth_flag": np.array([0, 0, 0, 0, 0, 2, 0, 0]),
        "photom_event_moon_flag": np.zeros(8),
        "photom_event_sun_flag": np.zeros(8),
        "photom_event_cosmic_flag": np.array([0, 0, 0, 0, 0, 0, 3, 0]),
    }


def _install_fake_dace(monkeypatch):
    calls = []

    class FakeCheops:
        @staticmethod
        def get_lightcurve(target, aperture="default", filters=None, output_format=None):
            calls.append((target, aperture, filters, output_format))
            data = _dace_data()
            if filters and "file_key" in filters:
                keep = np.isin(data["file_key"], filters["file_key"]["equal"])
                return {name: values[keep] for name, values in data.items()}
            return data

        @staticmethod
        def query_region(sky_coord, angle, output_format=None):
            return {
                "obj_id_catname": np.array(["Target A", "Target B"]),
                "file_key": np.array(["visit-a", "visit-b"]),
                "ra_deg": np.array((10.0, 10.01)),
                "dec_deg": np.array((20.0, 20.0)),
            }

    package = ModuleType("dace_query")
    cheops_module = ModuleType("dace_query.cheops")
    cheops_module.Cheops = FakeCheops
    monkeypatch.setitem(sys.modules, "dace_query", package)
    monkeypatch.setitem(sys.modules, "dace_query.cheops", cheops_module)
    return calls


def test_retrieve_cheops_light_curves_by_tic_splits_visits_and_normalizes(monkeypatch):
    calls = _install_fake_dace(monkeypatch)

    curves = miletos.retrieve_cheops_light_curves(42)

    assert calls[0][0] == "TIC 42"
    assert set(curves) == {"visit-a", "visit-b"}
    assert curves["visit-a"].shape == (3, 3)
    assert curves["visit-b"].shape == (2, 3)
    assert list(curves) == ["visit-b", "visit-a"]
    assert curves["visit-a"][0, 0] == pytest.approx(2459001.0)
    assert np.median(curves["visit-a"][:, 1]) == pytest.approx(1.0)
    assert np.all(curves["visit-a"][:, 2] > 0.0)


def test_retrieve_cheops_light_curves_resolves_nearest_coordinate_target(monkeypatch):
    calls = _install_fake_dace(monkeypatch)

    curves = miletos.retrieve_cheops_light_curves(ra_deg=10.0, dec_deg=20.0)

    assert calls[0][0] == "Target A"
    assert calls[0][2] == {"file_key": {"equal": ["visit-a"]}}
    assert set(curves) == {"visit-a"}


@pytest.mark.parametrize("column", [
    "obj_date_bjd_vect", "photom_flux_vect", "photom_flux_vect_err",
    "photom_event_saa_flag",
])
def test_retrieve_cheops_rejects_masked_archive_observations(monkeypatch, column):
    _install_fake_dace(monkeypatch)
    data = _dace_data()
    data[column] = np.ma.array(data[column], mask=[True] + [False] * 7)
    monkeypatch.setattr(sys.modules[__name__], "_dace_data", lambda: data)

    with pytest.raises(ValueError, match="masked observations"):
        miletos.retrieve_cheops_light_curves(42)


def test_cheops_column_accepts_unmasked_container():
    from miletos.cheops import _as_column

    values = np.arange(3)
    result = _as_column({"flux": np.ma.array(values, mask=False)}, "flux")
    np.testing.assert_array_equal(result, values)


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"ra_deg": 10.0},
        {"target": "WASP-8", "ra_deg": 10.0, "dec_deg": 20.0},
    ],
)
def test_retrieve_cheops_light_curves_validates_target_arguments(arguments):
    with pytest.raises(ValueError):
        miletos.retrieve_cheops_light_curves(**arguments)


def test_cheops_retrieval_keeps_only_latest_visit_revision():
    from importlib import import_module

    latest_visit_revisions = import_module("miletos.cheops")._latest_visit_revisions
    latest = latest_visit_revisions(np.array((
        "CH_PR100015_TG010401_V0200",
        "CH_PR100015_TG010401_V0300",
        "CH_PR100015_TG010701_V0200",
    )))

    assert latest == {"CH_PR100015_TG010401_V0300", "CH_PR100015_TG010701_V0200"}