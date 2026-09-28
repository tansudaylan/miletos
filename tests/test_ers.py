from pathlib import Path
from types import SimpleNamespace

from miletos import ers


def test_wasp39_reproduction_routes_all_plots_through_pipeline(monkeypatch, tmp_path):
    result = SimpleNamespace(wavelength_microns=[3.0, 4.0])
    calls = []
    monkeypatch.setattr(ers, 'analyze_wasp39_ers_g395h', lambda refresh_data: result)
    monkeypatch.setattr(ers, 'get_repository_path', lambda: tmp_path)
    monkeypatch.setattr(
        ers.visualization,
        'plot_wasp39_ers_white_light_curves',
        lambda value, path: calls.append(('white_light', value, path)),
    )
    for function_name, label in (
        ('plot_wasp39_ers_detector_motion', 'detector_motion'),
        ('plot_wasp39_ers_spectroscopic_detrending', 'spectroscopic_detrending'),
        ('plot_wasp39_ers_corrected_light_curve_map', 'corrected_light_curve_map'),
        ('plot_wasp39_ers_light_curve_precision', 'light_curve_precision'),
    ):
        monkeypatch.setattr(
            ers.visualization,
            function_name,
            lambda value, path, label=label: calls.append((label, value, path)),
        )
    monkeypatch.setattr(
        ers.visualization,
        'plot_wasp39_ers_transmission_spectrum',
        lambda value, path: calls.append(('transmission_spectrum', value, path)),
    )

    returned = ers.run_wasp39_ers_g395h_reproduction(typefileplot='pdf')

    output_path = tmp_path / 'examples' / 'WASP-39b' / 'visuals'
    assert returned is result
    assert calls == [
        ('white_light', result, output_path / 'wasp39_ers_g395h_white_light.pdf'),
        (
            'detector_motion',
            result,
            output_path / 'wasp39_ers_g395h_detector_motion.pdf',
        ),
        (
            'spectroscopic_detrending',
            result,
            output_path / 'wasp39_ers_g395h_spectroscopic_detrending.pdf',
        ),
        (
            'corrected_light_curve_map',
            result,
            output_path / 'wasp39_ers_g395h_corrected_light_curve_map.pdf',
        ),
        (
            'light_curve_precision',
            result,
            output_path / 'wasp39_ers_g395h_light_curve_precision.pdf',
        ),
        (
            'transmission_spectrum',
            result,
            output_path / 'wasp39_ers_g395h_transmission_spectrum.pdf',
        ),
    ]


def test_wasp39_analysis_module_does_not_render_plots():
    source = open(ers.__file__, encoding='utf-8').read()
    assert 'matplotlib' not in source
    assert '.savefig(' not in source


def test_wasp39_entry_point_only_invokes_pipeline():
    example_path = Path(__file__).parents[1] / 'examples' / 'WASP-39b' / 'run.py'
    source = example_path.read_text()
    assert 'run_wasp39_ers_g395h_reproduction' in source
    assert 'matplotlib' not in source
    assert '.savefig(' not in source