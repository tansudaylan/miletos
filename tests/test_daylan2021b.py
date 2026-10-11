from pathlib import Path

import numpy as np
import pytest

from miletos import daylan2021b


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
    assert captured['boolfitt'] is False
    assert Path(captured['pathtarg']) == tmp_path / 'examples' / 'WASP-121b'


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

    assert any('published 3012' in line for line in summary)
    assert any('published 2022' in line for line in summary)