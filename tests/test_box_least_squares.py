import numpy as np
import pytest

import miletos


def test_box_least_squares_recovers_period_and_depth():
    time_days = np.arange(0.0, 20.0, 0.02)  # [day]
    injected_period_days = 2.5  # [day]
    phase_days = (time_days - 0.4 + injected_period_days / 2.0) % injected_period_days
    phase_days -= injected_period_days / 2.0
    flux = np.ones_like(time_days)
    flux[np.abs(phase_days) < 0.06] -= 0.01
    flux += np.random.default_rng(7).normal(0.0, 0.001, time_days.size)
    uncertainty = np.full_like(time_days, 0.001)

    result = miletos.search_box_least_squares(
        time_days,
        flux,
        uncertainty,
        duration_days=[0.10, 0.12, 0.14],  # [day]
        minimum_period_days=2.0,  # [day]
        maximum_period_days=3.0,  # [day]
    )

    assert result["period_days"] == pytest.approx(injected_period_days, abs=0.02)
    assert result["depth"] == pytest.approx(0.01, abs=0.001)
    assert result["depth_snr"] > 20.0
    assert result["duration_days"] == pytest.approx(0.12, abs=0.02)


@pytest.mark.parametrize(
    "arguments",
    [
        {"time_days": [0.0, 1.0, 1.0]},
        {"flux_uncertainty": [0.1, 0.0, 0.1]},
        {"duration_days": [0.0]},
    ],
)
def test_box_least_squares_rejects_invalid_inputs(arguments):
    values = {
        "time_days": np.array([0.0, 1.0, 2.0]),  # [day]
        "flux": np.ones(3),
        "flux_uncertainty": np.full(3, 0.1),
        "duration_days": [0.2],  # [day]
        "minimum_period_days": 0.5,  # [day]
        "maximum_period_days": 1.0,  # [day]
    }
    values.update(arguments)

    with pytest.raises(ValueError):
        miletos.search_box_least_squares(**values)


def test_srch_boxsperi_astropy_uses_the_miletos_bls_api(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import miletos.box_least_squares as box_search
    from miletos.main import srch_boxsperi

    periodogram = SimpleNamespace(
        period=np.array([2.4, 2.5]),
        power=np.array([10.0, 30.0]),
        depth=np.array([0.008, 0.01]),
        depth_snr=np.array([20.0, 40.0]),
    )
    calls = []

    def fake_search(time_days, flux, flux_uncertainty, **kwargs):
        calls.append((time_days, flux, flux_uncertainty, kwargs))
        return {
            "period_days": 2.5,
            "transit_time_days": 0.4,
            "duration_days": 0.1,
            "depth": 0.01,
            "depth_snr": 40.0,
            "periodogram": periodogram,
        }

    monkeypatch.setattr(box_search, "search_box_least_squares", fake_search)
    time_days = np.arange(0.0, 20.0, 0.2)  # [day]
    arry = np.column_stack(
        (time_days, np.ones_like(time_days), np.full_like(time_days, 0.001))
    )
    result = srch_boxsperi(
        arry,
        typecalc="astropy",
        minmperi=2.0,  # [day]
        maxmperi=3.0,  # [day]
        pathdata=f"{tmp_path}/",
        boolprocmult=False,
        boolchecrebn=False,
        booldiag=False,
        typeverb=-1,
    )

    assert len(calls) == 1
    assert calls[0][3]["minimum_period_days"] == pytest.approx(2.0)
    assert calls[0][3]["maximum_period_days"] == pytest.approx(3.0)
    assert result["peri"].tolist() == [2.5]
    assert result["dura"][0] == pytest.approx(2.4)
    assert result["ampl"][0] == pytest.approx(10.0)
    assert result["s2nr"][0] == pytest.approx(40.0)