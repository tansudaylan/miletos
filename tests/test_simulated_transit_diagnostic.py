import matplotlib.image as mpimg
import numpy as np

from miletos.diagnostics import (
    run_simulated_jwst_transit_diagnostic,
    run_simulated_transit_diagnostic,
)


def test_simulated_transit_diagnostic_writes_pipeline_figure(tmp_path, capsys):
    output_path = tmp_path / "simulated_transit_diagnostic.png"

    result = run_simulated_transit_diagnostic(output_path)

    image = mpimg.imread(output_path)
    assert output_path.is_file()
    assert result.time_days.shape == result.observed_flux.shape
    assert result.time_days.size > 3_000
    assert result.binned_phase.size == 65
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
    assert f"Writing to {output_path}..." in capsys.readouterr().out


def test_simulated_jwst_diagnostic_remains_supported(tmp_path, capsys):
    output_path = tmp_path / 'simulated_jwst_transit.pdf'

    result = run_simulated_jwst_transit_diagnostic(output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 1_000
    assert result.observed_flux.shape == (
        result.time_days.size,
        result.wavelength_microns.size,
    )
    assert np.ptp(result.true_depth_ppm) > 500.0
    assert f'Writing to {output_path}...' in capsys.readouterr().out