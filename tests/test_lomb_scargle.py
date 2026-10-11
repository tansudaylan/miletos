import numpy as np
import pytest

from miletos.main import exec_lspe


def test_exec_lspe_returns_frequency_and_power_arrays():
    time_days = np.linspace(0.0, 20.0, 201)  # [day]
    injected_frequency_per_day = 0.25  # [1/day]
    flux = 1.0 + 0.01 * np.sin(2 * np.pi * injected_frequency_per_day * time_days)
    light_curve = np.column_stack((time_days, flux))

    result = exec_lspe(
        light_curve,
        minmfreq=0.1,  # [1/day]
        maxmfreq=0.5,  # [1/day]
        factosam=100,
        typeverb=-1,
    )

    assert result["freq"].shape == result["powr"].shape
    assert np.all(np.diff(result["freq"]) > 0)
    peak_index = int(np.argmax(result["powr"]))
    assert result["freq"][peak_index] == pytest.approx(injected_frequency_per_day)
    assert result["perimpow"] == pytest.approx(1 / result["freq"][peak_index])


def test_exec_lspe_returns_frequency_and_power_from_cache(tmp_path):
    frequency_per_day = np.array((0.1, 0.2, 0.3))
    power = np.array((0.1, 0.8, 0.2))
    cached_periodogram = np.column_stack((1 / frequency_per_day, power))
    np.savetxt(tmp_path / "LSPeriodogram_toy.csv", cached_periodogram, delimiter=",")

    result = exec_lspe(
        np.column_stack((np.arange(4), np.ones(4))),
        pathdata=f"{tmp_path}/",
        strgextn="toy",
        typeverb=-1,
    )

    np.testing.assert_allclose(result["freq"], frequency_per_day)
    np.testing.assert_allclose(result["powr"], power)
