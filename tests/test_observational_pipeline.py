from pathlib import Path

import matplotlib as mpl

from miletos import ers, toi1233
from miletos.pipeline import run_observational_pipeline


def test_observational_pipeline_owns_analysis_style_and_outputs(tmp_path):
    observations = []
    result = object()

    def analyze(refresh_data):
        observations.append(('analysis', refresh_data, mpl.rcParams['axes.facecolor']))
        return result

    def plot(value, path):
        observations.append(('plot', value, path, mpl.rcParams['axes.facecolor']))
        return path

    returned = run_observational_pipeline(
        analyzer=analyze,
        analysis_kwargs={'refresh_data': True},
        output_path=tmp_path,
        typefileplot='pdf',
        plot_products=(('diagnostic', plot),),
    )

    assert returned is result
    assert observations == [
        ('analysis', True, 'white'),
        ('plot', result, tmp_path / 'diagnostic.pdf', 'white'),
    ]


def test_wasp39_and_toi1233_use_the_same_observational_pipeline():
    assert ers.run_observational_pipeline is run_observational_pipeline
    assert toi1233.run_observational_pipeline is run_observational_pipeline


def test_toi1233_adapter_passes_shared_plot_configuration(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(toi1233, 'get_repository_path', lambda: tmp_path)
    monkeypatch.setattr(toi1233, '_load_toi1233_time_series', lambda: {'Raw': []})
    monkeypatch.setattr(
        toi1233.main,
        'init',
        lambda **kwargs: captured.update(kwargs) or {'strgtarg': 'TOI-1233'},
    )

    result = toi1233.analyze_toi1233(typefileplot='pdf')

    assert result == {'strgtarg': 'TOI-1233'}
    assert captured['typefileplot'] == 'pdf'
    assert captured['typeplotback'] == 'white'
    assert Path(captured['pathtarg']) == tmp_path / 'examples' / 'TOI-1233'