from pathlib import Path

import numpy as np
import pytest

from miletos import daylan2021b


def test_sector7_sap_loader_corrects_cached_time_offset_and_detrends_orbits(tmp_path):
    example_path = tmp_path / 'examples' / 'WASP-121b'
    data_path = example_path / 'data'
    data_path.mkdir(parents=True)
    cadences_per_day = 720
    first_orbit = 2458491.0 + np.arange(5 * cadences_per_day) / cadences_per_day
    second_orbit = 2458498.0 + np.arange(5 * cadences_per_day) / cadences_per_day
    time = np.concatenate((first_orbit, second_orbit))
    baselines = []
    for start, orbit_time in ((first_orbit[0], first_orbit), (second_orbit[0], second_orbit)):
        coordinate = (orbit_time - start) / 5.0
        baselines.append(1.0 + 0.002 * coordinate + 0.003 * coordinate**2 - 0.004 * coordinate**3)
    flux = np.concatenate(baselines)
    uncertainty = np.full(time.size, 1e-4)
    np.savetxt(
        data_path / 'TESS_SAP_FLUX_Sector07.csv',
        np.column_stack((time + daylan2021b.SAP_TIME_OFFSET, flux, uncertainty)),
        delimiter=',',
    )

    photometry = daylan2021b._read_sector7_sap(example_path)
    corrected_flux, corrected_uncertainty = daylan2021b._detrend_sector7_sap(*photometry.T)

    assert photometry[:, 0] == pytest.approx(time)
    assert corrected_flux == pytest.approx(np.ones_like(flux), abs=1e-8)
    assert corrected_uncertainty == pytest.approx(uncertainty / flux, rel=1e-8)


def test_sector7_sap_loader_preserves_miletos_bjd_export(tmp_path):
    data_path = tmp_path / 'examples' / 'WASP-121b' / 'data'
    data_path.mkdir(parents=True)
    time = np.array([2458491.0, 2458491.01])
    np.savetxt(
        data_path / 'TESS_SAP_FLUX_Sector07.csv',
        np.column_stack((time, [1.0, 1.0], [0.1, 0.1])),
        delimiter=',',
    )

    photometry = daylan2021b._read_sector7_sap(data_path.parent)

    assert photometry[:, 0] == pytest.approx(time)


def test_daylan2021b_uses_sector_seven_miletos_model(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(daylan2021b, 'get_repository_path', lambda: tmp_path)
    monkeypatch.setattr(
        daylan2021b.main,
        'init',
        lambda **configuration: captured.update(configuration)
        or {'strgtarg': 'WASP-121b-Daylan2021b'},
    )

    result = daylan2021b.run_daylan2021b_reproduction(fit=False)

    assert result == {'strgtarg': 'WASP-121b-Daylan2021b'}
    assert captured['listtsecsele'] == [7]
    assert captured['strgmast'] == 'WASP-121'
    assert captured['dictfitt']['typemodl'] == 'PlanetarySystemEmittingCompanion'
    assert captured['typepriocomp'] == 'inpt'
    assert captured['pericompprio'] == [daylan2021b.PERI_PUBL]
    assert captured['boolfitt'] is False
    assert Path(captured['pathtarg']) == tmp_path / 'examples' / 'WASP-121b'


def test_daylan2021b_uses_cached_sap_for_observation_plots(monkeypatch, tmp_path):
    example_path = tmp_path / 'examples' / 'WASP-121b'
    data_path = example_path / 'data'
    data_path.mkdir(parents=True)
    time = np.array([2458491.0, 2458491.01])
    photometry = np.column_stack((time + daylan2021b.SAP_TIME_OFFSET, [1.0, 1.0], [0.1, 0.1]))
    np.savetxt(data_path / 'TESS_SAP_FLUX_Sector07.csv', photometry, delimiter=',')
    captured = {}
    monkeypatch.setattr(daylan2021b, 'get_repository_path', lambda: tmp_path)
    monkeypatch.setattr(
        daylan2021b.main,
        'init',
        lambda **configuration: captured.update(configuration) or {'strgtarg': 'WASP-121b'},
    )

    daylan2021b.run_daylan2021b_reproduction(fit=False)

    assert captured['liststrgtypedata'] == [['inpt'], []]
    input_cube = captured['listarrytser']['Raw'][0][0][0]
    assert input_cube[:, 0, 0] == pytest.approx(time)


def test_daylan2021b_rejects_unknown_figure_format(monkeypatch):
    monkeypatch.setattr(daylan2021b.main, 'init', lambda **kwargs: pytest.fail('pipeline must not run'))

    with pytest.raises(ValueError, match="typefileplot must be 'png' or 'pdf'"):
        daylan2021b.run_daylan2021b_reproduction(typefileplot='svg')


def test_daylan2021b_summary_reports_published_temperatures():
    samples = np.zeros((16, len(daylan2021b.NAMES)))
    temperatures = {
        'day': np.full(16, 3010.0),
        'night': np.full(16, 2000.0),
    }

    summary = daylan2021b.retr_summary_daylan2021b(samples, temperatures)

    assert any('published 418 +/- 34 ppm' in line for line in summary)
    assert any('published 65 (+40 / -37) ppm' in line for line in summary)
    assert any('published -6.2 +/- 4.0 deg' in line for line in summary)
    assert any('published 3012' in line for line in summary)
    assert any('published 2022' in line for line in summary)