import os
import types

import pytest

import numpy as np

from miletos.visualization import plot_binned_rms, plot_work_tser, retr_compmodl_style, retr_resi_series, retr_stdvresi_series, setp_dictmodl_sample


class DummyGdat:
    pass


def test_plot_work_tser_forwards_standard_arguments():
    gdat = DummyGdat()
    gdat.pathvisutarg = '/tmp/visu/'
    gdat.timeoffs = 2457000.0
    gdat.boolwritover = False
    gdat.boolbrekmodl = True

    calls = {}

    def fake_plot_tser(pathvisu, **kwargs):
        calls['pathvisu'] = pathvisu
        calls['kwargs'] = kwargs
        return 'plot.png'

    pathplot = plot_work_tser(
        fake_plot_tser,
        gdat,
        timedata=[1, 2],
        tserdata=[3, 4],
        strgextn='Demo',
        strgtitl='Title',
        dictmodl={'model': 1},
        lablyaxi='Residual relative flux',
        booldiag=True,
    )

    assert pathplot == 'plot.png'
    assert calls['pathvisu'] == '/tmp/visu/'
    assert calls['kwargs']['timedata'] == [1, 2]
    assert calls['kwargs']['tserdata'] == [3, 4]
    assert calls['kwargs']['timeoffs'] == 2457000.0
    assert calls['kwargs']['strgextn'] == 'Demo'
    assert calls['kwargs']['strgtitl'] == 'Title'
    assert calls['kwargs']['boolwritover'] is False
    assert calls['kwargs']['boolbrekmodl'] is True
    assert calls['kwargs']['dictmodl'] == {'model': 1}
    assert calls['kwargs']['lablyaxi'] == 'Residual relative flux'
    assert calls['kwargs']['booldiag'] is True


def test_plot_binned_rms_writes_plot(tmp_path):
    gdat = DummyGdat()
    gdat.figrsizeydob = (4, 3)
    gdat.cadetimeplot = 0.5
    gdat.typeverb = 0

    path = str(tmp_path / 'stdvrebn.png')
    delt = [0.5, 1.0, 2.0]
    stdvresi = [1e-6, 2e-6, 3e-6]

    pathout = plot_binned_rms(gdat, path, delt, stdvresi)

    assert pathout == path
    assert os.path.exists(path)


def test_retr_compmodl_style_returns_expected_pairs():
    assert retr_compmodl_style('Total') == ('Total Model', 'b')
    assert retr_compmodl_style('Baseline') == ('Baseline', 'orange')
    assert retr_compmodl_style('Transit') == ('Transit', 'r')
    assert retr_compmodl_style('StarFlaring') == ('Flares', 'g')
    assert retr_compmodl_style('excs') == ('Excess', 'olive')


def test_setp_dictmodl_sample_sets_plot_metadata():
    dictmodl = {}

    setp_dictmodl_sample(dictmodl, 'sample', [1, 2], [3, 4], 'Model', 'b', 0.2, booldiag=True)

    assert dictmodl['sample']['tser'] == [1, 2]
    assert dictmodl['sample']['time'] == [3, 4]
    assert dictmodl['sample']['labl'] == 'Model'
    assert dictmodl['sample']['colr'] == 'b'
    assert dictmodl['sample']['alph'] == 0.2


def test_setp_dictmodl_sample_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        setp_dictmodl_sample({}, 'sample', [1], [1, 2], 'Model', 'b', 0.2, booldiag=True)


def test_retr_resi_series_for_sampled_full_model():
    gdat = DummyGdat()
    gdat.typeinfe = 'samp'
    gdat.fitt = types.SimpleNamespace(typemodlenerfitt='full')
    gdat.dictsamp = {'resitest': np.array([[[1.0, 3.0], [5.0, 7.0]], [[3.0, 5.0], [7.0, 9.0]]])}
    gmod = types.SimpleNamespace()

    tserdatatemp = retr_resi_series(gdat, gmod, 'test', 1)

    assert np.allclose(tserdatatemp, np.array([4.0, 8.0]))


def test_retr_resi_series_for_mlik_sparse_model():
    gdat = DummyGdat()
    gdat.typeinfe = 'mlik'
    gdat.fitt = types.SimpleNamespace(typemodlenerfitt='sparse')
    gmod = types.SimpleNamespace(listdictmlik=[{'resitest': np.array([[1.0], [2.0], [3.0]])}])

    tserdatatemp = retr_resi_series(gdat, gmod, 'test', 0)

    assert np.allclose(tserdatatemp, np.array([1.0, 2.0, 3.0]))


def test_retr_stdvresi_series_for_sampled_sparse_model():
    gdat = DummyGdat()
    gdat.typeinfe = 'samp'
    gdat.fitt = types.SimpleNamespace(typemodlenerfitt='sparse')
    gmod = types.SimpleNamespace(listdictsamp=[{'stdvresitest': np.array([[[1.0], [2.0]], [[3.0], [4.0]]])}])

    stdvresi = retr_stdvresi_series(gdat, gmod, 'test', 0)

    assert np.allclose(stdvresi, np.array([2.0, 3.0]))


def test_retr_stdvresi_series_for_mlik_full_model():
    gdat = DummyGdat()
    gdat.typeinfe = 'mlik'
    gdat.fitt = types.SimpleNamespace(typemodlenerfitt='full')
    gdat.dictmlik = {'stdvresitest': np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])}
    gmod = types.SimpleNamespace()

    stdvresi = retr_stdvresi_series(gdat, gmod, 'test', 1)

    assert np.allclose(stdvresi, np.array([10.0, 20.0, 30.0]))