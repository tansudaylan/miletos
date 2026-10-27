"""Production photometry must work with Lightkurve unavailable."""

import ast
import builtins
from pathlib import Path
import subprocess
import sys
import textwrap
import tomllib

import numpy as np
import pytest
from astropy import units
from astropy.io import fits
from astropy.time import Time

from miletos.light_curve import LightCurve
from miletos.photometry import prepare_light_curve_collection, read_tess_light_curve


def test_lightkurve_is_not_an_install_dependency_or_production_import():
    root = Path(__file__).resolve().parents[1]
    metadata_path = root / 'pyproject.toml'
    print(f'Reading from {metadata_path}...')
    metadata = tomllib.loads(metadata_path.read_text())['project']
    dependencies = metadata['dependencies'] + [item for values in
                    metadata['optional-dependencies'].values() for item in values]
    assert not any('lightkurve' in dependency.lower() for dependency in dependencies)
    for filename in (root / 'miletos').rglob('*.py'):
        if filename.name == 'benchmarks.py':
            continue
        print(f'Reading from {filename}...')
        for node in ast.walk(ast.parse(filename.read_text())):
            if isinstance(node, ast.Import):
                assert not any(alias.name.split('.')[0] == 'lightkurve' for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or '').split('.')[0] != 'lightkurve'
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {'to_periodogram', 'search_lightcurve'}


@pytest.mark.parametrize('missing_errors', [False, True])
def test_native_preprocessing_retains_normalized_errors_and_absolute_epochs(missing_errors):
    time = Time(2459000 + np.arange(600) / 50, format='jd', scale='tdb')
    errors = np.full(600, np.nan if missing_errors else 2.)
    original = LightCurve(time, np.full(600, 100.) * units.electron / units.s,
                          errors * units.electron / units.s)
    result = prepare_light_curve_collection([original])
    np.testing.assert_array_equal(result.time, time)
    np.testing.assert_array_equal(result.flux.value, np.ones(600))
    np.testing.assert_allclose(result.flux_err.value, errors / 100, equal_nan=True)
    np.testing.assert_array_equal(original.flux.value, np.full(600, 100.))


def test_native_folding_sorts_data_and_binning_propagates_errors():
    curve = LightCurve(np.array([0.12, 0.22, 0.6, 0.7]), np.array([1., 3., 5., 7.]),
                       np.full(4, 2.))
    folded = curve.fold(1., 0.)
    np.testing.assert_allclose(folded.time.to_value(units.day), [-0.4, -0.3, 0.12, 0.22])
    np.testing.assert_array_equal(folded.flux.value, [5., 7., 1., 3.])
    binned = folded.bin(0.25)
    np.testing.assert_allclose(binned.flux.value, [6., 2.])
    np.testing.assert_allclose(binned.flux_err.value, np.sqrt(2.))


def test_native_fits_reader_preserves_quality_selection_and_bjd_reference(tmp_path):
    filename = tmp_path / 'tess_lc.fits'
    columns = [fits.Column(name='TIME', format='D', array=[1., 2., 3., np.nan]),
               fits.Column(name='PDCSAP_FLUX', format='D', array=[10., 20., 30., 40.]),
               fits.Column(name='PDCSAP_FLUX_ERR', format='D', array=[1., 2., 3., 4.]),
               fits.Column(name='QUALITY', format='J', array=[0, 1, 256, 0])]
    table = fits.BinTableHDU.from_columns(columns)
    table.header['BJDREFI'] = 2457000
    print(f'Writing to {filename}...')
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(filename)
    curve = read_tess_light_curve(filename)
    np.testing.assert_array_equal(curve.time.jd, [2457001., 2457003.])
    np.testing.assert_array_equal(curve.flux.value, [10., 30.])
    np.testing.assert_array_equal(curve.flux_err.value, [1., 3.])


def test_complete_native_workflow_with_lightkurve_imports_blocked():
    code = textwrap.dedent('''
        import builtins
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name == 'lightkurve' or name.startswith('lightkurve.'):
                raise AssertionError('Production attempted to import Lightkurve')
            return original_import(name, *args, **kwargs)
        builtins.__import__ = guarded_import
        import setuptools
        import numpy as np
        from astropy import units
        from astropy.time import Time
        import miletos.benchmarks
        from miletos.light_curve import LightCurve
        from miletos.photometry import prepare_light_curve_collection
        from miletos.box_least_squares import fixed_grid_box_least_squares, fold_lightkurve
        from miletos.simulation import inject_batman_transit
        time = Time(2459000 + np.arange(600) / 50, format='jd', scale='tdb')
        curve = prepare_light_curve_collection([LightCurve(time, np.ones(600), np.full(600, .001))])
        injected = inject_batman_transit(123, curve, curve.time.value,
            radius_star=.01, radius_planet=.005, period=2.)
        result = fixed_grid_box_least_squares(injected)
        folded = fold_lightkurve(injected, result.period_at_max_power,
            result.transit_time_at_max_power, bin=True, time_bin_size=.01)
        assert len(folded) > 0 and len(result.period) == 2048
    ''')
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)