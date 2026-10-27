import numpy as np
import pytest

import miletos


def test_fixed_grid_bls_calls_astropy_without_external_curve_methods(monkeypatch):
    from unittest.mock import Mock
    from astropy import units
    from astropy.timeseries import BoxLeastSquares
    import miletos.box_least_squares as shared
    from miletos.light_curve import LightCurve
    from miletos.box_least_squares import fixed_grid_box_least_squares

    expected = BoxLeastSquares(np.arange(6) * units.day, np.ones(6)).power(
        [2.] * units.day, [.1] * units.day)
    power = Mock(return_value=expected)
    model = Mock(return_value=Mock(power=power))
    monkeypatch.setattr(shared, 'BoxLeastSquares', model)
    curve = LightCurve([0., 1., 2.], [1., .98, 1.], [.001, .001, .001])
    assert fixed_grid_box_least_squares(curve) is expected
    periods = np.logspace(0, 1, 2048)  # [day]
    np.testing.assert_array_equal(power.call_args.args[0], periods)
    np.testing.assert_array_equal(power.call_args.args[1], periods * 0.01)


def test_lightkurve_candidates_retain_raw_first_power_and_aligned_indices():
    from types import SimpleNamespace
    from astropy import units
    from scipy.signal import medfilt
    from miletos.box_least_squares import select_lightkurve_bls_candidates

    periods = np.geomspace(1, 10, 2048) * units.day
    powers = np.full(2048, 2.)
    indices = [20, 500, 1000, 1500]
    powers[indices] += [10., 9., 8., 7.]
    result = select_lightkurve_bls_candidates(SimpleNamespace(
        period=periods, power=powers * units.dimensionless_unscaled,
    ))
    assert result['candidate_indices'] == indices
    assert result['best_index'] == indices[0]
    np.testing.assert_array_equal(result['periods'], periods.value[indices])
    np.testing.assert_array_equal(result['powers'], [12., 9., 8., 7.])
    np.testing.assert_array_equal(result['detrended_power'], powers - medfilt(powers, 101))


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


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_animate_box_least_squares_search_writes_gif(tmp_path, monkeypatch, typeplotback):
    from PIL import Image
    import matplotlib.animation as animation_module
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.text import Text
    from tdpy.plotting import plot_style

    figures = []
    original_animation = animation_module.FuncAnimation

    def capture_animation(figure, *args, **kwargs):
        figures.append(figure)
        return original_animation(figure, *args, **kwargs)

    monkeypatch.setattr(animation_module, 'FuncAnimation', capture_animation)
    time_days = np.arange(300, dtype=float) * 0.02
    phase_days = (time_days - 0.4 + 1.2) % 2.4 - 1.2
    flux = 1.0 - 0.01 * (np.abs(phase_days) < 0.08)
    flux += np.random.default_rng(2).normal(0.0, 0.001, time_days.size)
    data = np.column_stack((time_days, flux, np.full(time_days.size, 0.001)))
    original_data = data.copy()
    path = tmp_path / "bls_scan.gif"

    with plot_style('white'):
        original_settings = dict(plt.rcParams)
        result = miletos.animate_box_least_squares_search(
            {"Sector 3": data}, path, minimum_period_days=1.5, maximum_period_days=3.0,
            duration_days=[0.08, 0.12], number_of_periods=8, maximum_points=300,
            frames_per_second=4, typeplotback=typeplotback,
        )
        assert dict(plt.rcParams) == original_settings

    assert result["path"] == path
    assert result["summary"]["Sector 3"]["period_days"] == pytest.approx(2.36, abs=0.15)
    figure = figures[0]
    assert [axis.get_xlabel() for axis in figure.axes] == ["Trial period [day]", "Orbital phase [1]"]
    assert [axis.get_ylabel() for axis in figure.axes] == ["BLS depth SNR [1]", "Relative flux [1]"]
    assert len(figure.legends) == 1
    assert all(text.get_fontsize() == 12 for text in figure.findobj(match=Text))
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    legend_bounds = figure.legends[0].get_window_extent(canvas.get_renderer())
    for axis in figure.axes:
        assert not any(line.get_visible() for line in axis.get_xgridlines() + axis.get_ygridlines())
        assert not legend_bounds.overlaps(axis.get_window_extent(canvas.get_renderer()))
    expected_data_color = "white" if typeplotback == 'dark' else "black"
    expected_model_color = "#FFB000" if typeplotback == 'dark' else "#D55E00"
    expected_alias_color = "#FF7A72" if typeplotback == 'dark' else "#A51C30"
    assert figure.axes[0].lines[1].get_color() == expected_data_color
    assert figure.axes[0].lines[0].get_color() == expected_alias_color
    assert figure.axes[1].lines[0].get_color() == expected_model_color
    with Image.open(path) as image:
        assert image.n_frames == 8
        for index in range(image.n_frames):
            image.seek(index)
            pixel = image.convert('RGB').getpixel((0, 0))
            expected = (0, 0, 0) if typeplotback == 'dark' else (255, 255, 255)
            assert pixel == expected
    np.testing.assert_array_equal(data, original_data)


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_lightkurve_bls_plot_labels_units_and_saves_pdf(tmp_path, monkeypatch, typeplotback):
    from types import SimpleNamespace
    from importlib import import_module

    from astropy import units
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.text import Text
    from tdpy.plotting import plot_background_colors, save_figure

    figures = []
    box_search = import_module('miletos.box_least_squares')

    def capture_figure(figure, *args, **kwargs):
        figures.append(figure)
        return save_figure(figure, *args, **kwargs)

    monkeypatch.setattr(box_search, 'save_figure', capture_figure)
    periods = np.geomspace(1.0, 10.0, 2048) * units.day
    powers = np.ones(2048) * units.dimensionless_unscaled
    powers[500] = 4.0 * units.dimensionless_unscaled
    results = SimpleNamespace(
        period=periods,
        power=powers,
        transit_time=np.full(2048, 0.4) * units.day,
        duration=np.full(2048, 0.1) * units.day,
    )

    box_search.lightkurve_bls_results(
        results, folder=str(tmp_path), ID='target', plot='save',
        typeplotback=typeplotback, typefileplot='pdf'
    )

    assert (tmp_path / 'target_blsplot.pdf').is_file()
    figure = figures[0]
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    axis = figure.axes[0]
    assert axis.get_xlabel() == 'Period [day]'
    assert axis.get_ylabel() == 'BLS power [1]'
    assert axis.lines[0].get_color() == plot_background_colors(typeplotback)[1]
    assert axis.lines[0].get_linestyle() == 'None'
    assert axis.lines[1].get_color() == ('#FFB000' if typeplotback == 'dark' else '#D55E00')
    assert axis.lines[2].get_color() == ('#FF7A72' if typeplotback == 'dark' else '#A51417')
    assert axis.get_xgridlines()[0].get_visible() is False
    legend = figure.legends[0]
    assert [text.get_text() for text in legend.get_texts()].count('Integer aliases') == 1
    assert legend.get_frame().get_alpha() == 1.0
    assert all(text.get_fontsize() == 12 for text in figure.findobj(match=Text))
    assert not legend.get_window_extent(canvas.get_renderer()).overlaps(
        axis.get_window_extent(canvas.get_renderer())
    )


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_folded_light_curve_labels_time_and_normalized_flux(tmp_path, monkeypatch, typeplotback):
    from astropy import units
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.text import Text
    from miletos.box_least_squares import fold_lightkurve
    from tdpy.plotting import plot_background_colors
    from miletos.light_curve import LightCurve

    figures = []
    import miletos.box_least_squares as box_search
    original_apply_plot_theme = box_search.apply_plot_theme

    def capture_figure(figure, *args, **kwargs):
        figures.append(figure)
        return original_apply_plot_theme(figure, *args, **kwargs)

    monkeypatch.setattr(box_search, 'apply_plot_theme', capture_figure)
    light_curve = LightCurve(
        np.linspace(0.0, 5.0, 101) * units.day,
        (1.0 - 0.01 * (np.abs(np.linspace(-0.5, 0.5, 101)) < 0.05)) * units.dimensionless_unscaled,
    )
    fold_lightkurve(
        light_curve, 2.5 * units.day, 0.4 * units.day, plot='save', ID='target',
        folder=str(tmp_path), typeplotback=typeplotback, typefileplot='pdf',
    )
    assert (tmp_path / 'target_2.5_foldedlc.pdf').is_file()
    figure = figures[0]
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    assert figure.axes[0].get_xlabel() == 'Time from transit [day]'
    assert figure.axes[0].get_ylabel() == 'Normalized flux [1]'
    assert len(figure.legends) == 1
    np.testing.assert_allclose(
        figure.axes[0].collections[0].get_facecolors()[0],
        plt.matplotlib.colors.to_rgba(plot_background_colors(typeplotback)[1]),
    )
    assert all(text.get_fontsize() == 12 for text in figure.findobj(match=Text))
    legend_bounds = figure.legends[0].get_window_extent(canvas.get_renderer())
    assert not legend_bounds.overlaps(figure.axes[0].get_window_extent(canvas.get_renderer()))
    plt.close(figure)