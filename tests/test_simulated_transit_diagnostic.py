import matplotlib.image as mpimg
import numpy as np
import pytest

from miletos.diagnostics import (
    run_simulated_jwst_transit_diagnostic,
    run_simulated_transit_diagnostic,
)


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_simulated_transit_diagnostic_writes_pipeline_figure(tmp_path, capsys, monkeypatch, typeplotback):
    import tdpy

    save = tdpy.save_figure
    captured = []

    def check_folded_panel(figure, *args, **kwargs):
        axis = figure.axes[2]
        assert axis.get_xlim() == (-0.5, 0.5)
        measured, model = axis.lines[:2]
        assert np.min(measured.get_xdata()) < 0 < np.max(measured.get_xdata())
        assert np.min(model.get_xdata()) < 0 < np.max(model.get_xdata())
        captured.append((np.array(model.get_xdata()), np.array(model.get_ydata())))
        return save(figure, *args, **kwargs)

    monkeypatch.setattr(tdpy, "save_figure", check_folded_panel)
    output_path = tmp_path / "simulated_transit_diagnostic.png"

    result = run_simulated_transit_diagnostic(output_path, typeplotback=typeplotback)

    image = mpimg.imread(output_path)
    assert output_path.is_file()
    assert result.time_days.shape == result.observed_flux.shape
    assert result.time_days.size > 3_000
    np.testing.assert_array_equal(captured[0][0], result.phase)
    np.testing.assert_array_equal(captured[0][1], 1e6 * (result.folded_model_flux - 1.0))
    assert result.binned_phase.size == 65
    assert result.phase.min() <= -0.49
    assert result.phase.max() >= 0.49
    assert np.any((-0.08 < result.binned_phase) & (result.binned_phase < 0))
    assert np.any((0 < result.binned_phase) & (result.binned_phase < 0.08))
    assert np.any((result.phase < 0) & (result.folded_model_flux < 0.999))
    assert np.any((result.phase > 0) & (result.folded_model_flux < 0.999))
    assert np.isfinite(result.binned_flux).all()
    assert 5_000 < 1e6 * (1.0 - result.model_flux.min()) < 10_000
    assert abs(np.median(result.detrended_flux) - 1.0) < 100e-6
    assert image.shape[0] > 100
    assert image.shape[1] > 100
    assert image[..., :3].min() < 0.8
    np.testing.assert_array_equal(image[0, 0], (0, 0, 0, 1) if typeplotback == 'dark' else (1, 1, 1, 1))
    assert f"Writing to {output_path}..." in capsys.readouterr().out


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_simulated_transit_diagnostic_writes_bokeh_html(tmp_path, monkeypatch, typeplotback):
    pytest.importorskip("bokeh")
    from bokeh.models import Axis, Plot
    import tdpy.plotting as plotting

    save = plotting.save_bokeh_html

    def capture_and_save(figure, *args, **kwargs):
        result = save(figure, *args, **kwargs)
        background, foreground = plotting.plot_background_colors(typeplotback)
        assert all(plot.background_fill_color == background for plot in figure.select({'type': Plot}))
        assert all(axis.major_label_text_color == foreground for axis in figure.select({'type': Axis}))
        return result

    monkeypatch.setattr(plotting, 'save_bokeh_html', capture_and_save)
    output_path = tmp_path / "simulated_transit_diagnostic.html"
    result = run_simulated_transit_diagnostic(output_path, typeplotback=typeplotback)

    assert result.time_days.size > 3_000
    assert output_path.is_file()
    assert "Input, simulated" in output_path.read_text(encoding="utf-8")


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_simulated_jwst_diagnostic_remains_supported(tmp_path, capsys, typeplotback):
    output_path = tmp_path / 'simulated_jwst_transit.pdf'

    result = run_simulated_jwst_transit_diagnostic(output_path, typeplotback=typeplotback)

    assert output_path.is_file()
    assert output_path.stat().st_size > 1_000
    assert result.observed_flux.shape == (
        result.time_days.size,
        result.wavelength_microns.size,
    )
    assert np.ptp(result.true_depth_ppm) > 500.0
    assert f'Writing to {output_path}...' in capsys.readouterr().out