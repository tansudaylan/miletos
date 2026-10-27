from pathlib import Path
import json
import runpy
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


def write_qlp_fixture(tmp_path, tic_id=380907942, sector=30):
    from astropy.io import fits

    path = tmp_path / f'hlsp_qlp_tess_ffi_s{sector:04d}-{tic_id:016d}_tess_v01_llc.fits'
    times = np.linspace(2114.0, 2122.0, 73)
    flux = 1000.0 * (1.0 + 0.001 * np.exp(-0.5*((times-2118.0)/0.5)**2))
    columns = [fits.Column(name='TIME', format='D', array=times),
               fits.Column(name='SAP_FLUX', format='D', array=flux),
               fits.Column(name='KSPSAP_FLUX', format='D', array=np.full(73, 1000.0)),
               fits.Column(name='KSPSAP_FLUX_ERR', format='D', array=np.ones(73)),
               fits.Column(name='QUALITY', format='J', array=np.zeros(73, dtype=int))]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path)
    return path


@pytest.fixture
def example(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[1] / 'examples/TIC380907942/run.py'
    namespace = runpy.run_path(str(source))
    function = namespace['run_analysis']
    monkeypatch.setitem(function.__globals__, 'get_data_path', lambda: tmp_path / 'data')
    monkeypatch.setitem(function.__globals__, 'get_repository_path', lambda: tmp_path)
    return namespace


def test_six_cadence_summary_matches_stated_means_and_scatter(example):
    curve = pd.DataFrame({'time_btjd': np.arange(13.), 'flux': np.arange(13.)/100+1,
                          'flux_error': np.full(13, 0.01)})
    before = curve.copy(deep=True)
    result = example['six_cadence_summary'](curve)
    assert len(result) == 2
    np.testing.assert_array_equal(result['time_btjd'], [2.5, 8.5])
    np.testing.assert_allclose(result['flux'], [1.025, 1.085])
    np.testing.assert_allclose(result['flux_scatter'], np.std(np.arange(6.)/100, ddof=0))
    assert result['cadences_per_bin'].eq(6).all()
    pd.testing.assert_frame_equal(curve, before)


@pytest.mark.parametrize('time_tag', ['bad', '../20261026_000000', '20260230_000000'])
def test_invalid_time_tag_rejects_before_data_access(example, monkeypatch, time_tag):
    reader = Mock(side_effect=AssertionError('Invalid timestamp reached input reading'))
    monkeypatch.setitem(example['run_analysis'].__globals__, 'read_qlp_light_curve', reader)
    with pytest.raises(ValueError):
        example['run_analysis'](time_tag=time_tag)
    reader.assert_not_called()


def test_existing_output_is_preserved_before_data_access(example, monkeypatch, tmp_path):
    output = tmp_path / 'data/TIC380907942/sector30_qlp_20261026_000000.csv'
    output.parent.mkdir(parents=True)
    output.write_bytes(b'Existing observations')
    reader = Mock(side_effect=AssertionError('Collision reached input reading'))
    monkeypatch.setitem(example['run_analysis'].__globals__, 'read_qlp_light_curve', reader)
    with pytest.raises(FileExistsError):
        example['run_analysis'](time_tag='20261026_000000')
    assert output.read_bytes() == b'Existing observations'
    reader.assert_not_called()


@pytest.mark.parametrize('theme', ['white', 'dark'])
def test_observed_workflow_uses_reader_and_retains_raw_tail_and_scope(example, tmp_path, theme, monkeypatch):
    from PIL import Image

    run_globals = example['run_analysis'].__globals__
    original_save_figure = run_globals['save_figure']
    legend_labels = []

    def capture_legend(figure, *args, **kwargs):
        legend_labels.extend(text.get_text() for text in figure.legends[0].get_texts())
        return original_save_figure(figure, *args, **kwargs)

    monkeypatch.setitem(run_globals, 'save_figure', capture_legend)
    input_path = write_qlp_fixture(tmp_path)
    original = input_path.read_bytes()
    paths = example['run_analysis'](input_path=input_path, time_tag='20261026_000000', typeplotback=theme)

    raw = pd.read_csv(paths['raw'])
    binned = pd.read_csv(paths['binned'])
    manifest = json.loads(paths['manifest'].read_text())
    assert len(raw) == 73 and len(binned) == 12
    assert raw['flux'].eq(1.0).all()
    assert raw['aperture_flux'].max() > 1.0005
    assert binned['flux'].max() > 1.0005
    assert manifest['unbinned_tail_cadences'] == 1
    assert manifest['retained_cadences'] == 73
    assert manifest['paper_doi'] == '10.3847/1538-4357/ae25f8'
    assert 'No radial-velocity time series' in manifest['scope']
    assert 'ddof=0' in manifest['bin_error_definition']
    assert manifest['displayed_flux'].startswith('SAP_FLUX')
    assert 'no fabricated error bars' in manifest['aperture_uncertainties']
    assert 'QLP aperture flux' in legend_labels
    assert 'QLP detrended flux' in legend_labels
    assert paths['figure'].parent == tmp_path / 'examples/TIC380907942/visuals'
    assert input_path.read_bytes() == original
    with Image.open(paths['figure']) as image:
        assert image.width > 100 and image.height > 100
        assert image.convert('RGB').getpixel((0, 0)) == ((255, 255, 255) if theme == 'white' else (0, 0, 0))


@pytest.mark.parametrize('tic_id,sector', [(1, 30), (380907942, 29)])
def test_wrong_target_or_sector_rejects_before_publication(example, tmp_path, tic_id, sector):
    input_path = write_qlp_fixture(tmp_path, tic_id=tic_id, sector=sector)
    with pytest.raises(ValueError, match='TIC 380907942 Sector 30'):
        example['run_analysis'](input_path=input_path, time_tag='20261026_000000')
    assert not (tmp_path / 'data').exists()