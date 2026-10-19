from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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


@pytest.mark.parametrize('name', [
    'plot_wasp39_ers_white_light_curves', 'plot_wasp39_ers_detector_motion',
    'plot_wasp39_ers_spectroscopic_detrending', 'plot_wasp39_ers_corrected_light_curve_map',
    'plot_wasp39_ers_light_curve_precision', 'plot_wasp39_ers_transmission_spectrum',
])
@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_wasp39_renderers_theme_synthetic_arrays_without_modification(tmp_path, monkeypatch, name, typeplotback):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from PIL import Image
    from tdpy.plotting import plot_background_colors, plot_style

    time = 59791.0 + np.linspace(-0.1, 0.1, 20)  # [BJD day]
    wavelength = np.r_[np.linspace(2.8, 3.7, 6), np.linspace(4.05, 5.1, 6)]  # [micron]
    transit = np.tile(1 - 0.02 * np.exp(-np.linspace(-3, 3, time.size) ** 2), (wavelength.size, 1))
    systematics = np.tile(1 + np.linspace(-0.001, 0.001, time.size), (wavelength.size, 1))
    uncertainty = np.full_like(transit, 0.0001)
    result = SimpleNamespace(
        nrs1_time_bjd=time, nrs2_time_bjd=time, nrs1_flux=transit[0], nrs2_flux=transit[-1],
        nrs1_flux_uncertainty=uncertainty[0], nrs2_flux_uncertainty=uncertainty[-1],
        spectroscopic_time_bjd=time, spectroscopic_wavelength_microns=wavelength,
        detector_shift_x_pixels=np.linspace(-0.02, 0.02, time.size),
        detector_shift_y_pixels=np.linspace(0.01, -0.01, time.size),
        raw_spectroscopic_flux=systematics + transit - 1,
        raw_spectroscopic_flux_uncertainty=uncertainty,
        corrected_spectroscopic_flux=transit,
        corrected_spectroscopic_flux_uncertainty=uncertainty,
        spectroscopic_residual_flux=np.zeros_like(transit),
        systematic_model=systematics, transit_model=transit,
        photon_precision_ppm=np.full(wavelength.size, 400.0),
        data_precision_ppm=np.full(wavelength.size, 600.0),
        wavelength_microns=wavelength, bin_half_width_microns=np.full(wavelength.size, 0.02),
        transit_depth_ppm=np.linspace(22000, 22500, wavelength.size),
        transit_depth_uncertainty_ppm=np.full(wavelength.size, 100.0),
        model_transit_depth_ppm=np.linspace(22000, 22500, wavelength.size),
        model_residual_ppm=np.zeros(wavelength.size), reduced_chi_squared=1.0,
    )
    originals = {key: value.copy() for key, value in vars(result).items() if isinstance(value, np.ndarray)}
    background, foreground = plot_background_colors(typeplotback)
    save = plt.Figure.savefig

    def capture_and_save(figure, path, **kwargs):
        assert figure.get_facecolor() == to_rgba(background)
        assert all(axis.xaxis.label.get_color() == foreground for axis in figure.axes)
        return save(figure, path, **kwargs)

    monkeypatch.setattr(plt.Figure, 'savefig', capture_and_save)
    path = tmp_path / 'synthetic.png'
    with plot_style('white'):
        original_settings = dict(plt.rcParams)
        assert getattr(ers.visualization, name)(result, path, typeplotback=typeplotback) == path
        assert dict(plt.rcParams) == original_settings
    for key, value in originals.items():
        np.testing.assert_array_equal(getattr(result, key), value)
    with Image.open(path) as image:
        expected = tuple(int(channel * 255) for channel in to_rgba(background)[:3])
        assert image.convert('RGB').getpixel((0, 0)) == expected