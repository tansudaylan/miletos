import os
import types

import pytest

import numpy as np

from miletos.visualization import build_abundance_component_specs, build_albg_comparison_data, build_component_sample_dict, build_feature_pair_guides, build_feature_pair_panel_meta, build_feature_pair_population_render_plan, build_feature_pair_target_render_plan, build_helium_comparison_data, build_magnitude_population_plot_data, build_occurrence_highlights, build_occurrence_rate_data, build_pcur_post, build_period_ratio_highlights, build_period_ratio_resonances, build_population_feature_plot_config, build_population_merge_data, build_population_sort_plot_data, build_psii_kdeg_data, build_psii_summary_data, build_ptem_plot_data, build_spec_data_groups, build_spec_model_data, build_total_sample_dict, check_feature_pair_selected, plot_binned_rms, plot_work_tser, retr_compmodl_style, retr_pcur_binned_series, retr_pcur_component_overlay, retr_pcur_lablpara, retr_pcur_model_series, retr_pcur_raw_series, retr_pcur_sample_plot_data, retr_resi_series, retr_stdvresi_series, retr_modl_fine_series, setp_dictmodl_sample
from miletos.visualization import retr_lablinst_part, retr_lablcnfg_part, retr_summary_extn, retr_summary_title


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


def test_retr_summary_title_and_parts():
    gdat = DummyGdat()
    gdat.listlablinst = [['TESS', 'JWST']]
    gdat.lablcnfg = 'DemoCfg'
    gdat.labltarg = 'WASP-121'
    gdat.numbener = [2, 1]
    gdat.listener = {0: [4.5]}

    assert retr_lablinst_part(gdat, 0, 0) == ', TESS'
    assert retr_lablcnfg_part(gdat) == ', DemoCfg'
    assert retr_summary_title(gdat, 0, 0, 0) == 'WASP-121, TESS, DemoCfg, white'
    assert retr_summary_title(gdat, 0, 0, 1) == 'WASP-121, TESS, DemoCfg, 4.5 micron'


def test_retr_summary_title_single_band_no_labels():
    gdat = DummyGdat()
    gdat.listlablinst = [['']]
    gdat.lablcnfg = ''
    gdat.labltarg = 'Target'
    gdat.numbener = [1]
    gdat.listener = {0: []}

    assert retr_summary_title(gdat, 0, 0, 0) == 'Target'


def test_retr_summary_extn_appends_band_suffix_only_when_needed():
    gdat = DummyGdat()
    gdat.strgcnfg = 'Cfg'
    gdat.numbener = [1, 3]
    gdat.liststrgdatafittiter = ['ignored', '_band1']

    assert retr_summary_extn('PosteriorSamples', gdat, 1, 0) == 'PosteriorSamplesCfg'
    assert retr_summary_extn('PosteriorSamples', gdat, 1, 1) == 'PosteriorSamplesCfg_band1'


def test_retr_modl_fine_series_for_full_and_sparse_modes():
    gdat = DummyGdat()
    gdat.fitt = types.SimpleNamespace(typemodlenerfitt='full')
    gdat.dictsamp = {'Model_Fine_Total_demo': np.array([[[1.0, 10.0], [2.0, 20.0]]])}
    gmod = types.SimpleNamespace(listdictsamp=[{'Model_Fine_Total_demo': np.array([[[3.0], [4.0]]])}])

    assert np.allclose(retr_modl_fine_series(gdat, gmod, 'Total', 'demo', 0, 1), np.array([10.0, 20.0]))

    gdat.fitt = types.SimpleNamespace(typemodlenerfitt='sparse')
    assert np.allclose(retr_modl_fine_series(gdat, gmod, 'Total', 'demo', 0, 0), np.array([3.0, 4.0]))


def test_build_total_sample_dict_creates_standard_entries():
    gdat = DummyGdat()
    gdat.numbsampplot = 2
    gdat.booldiag = True
    gdat.fitt = types.SimpleNamespace(typemodlenerfitt='full')
    gdat.dictsamp = {'Model_Fine_Total_demo': np.array([[[1.0], [2.0]], [[3.0], [4.0]]])}
    gmod = types.SimpleNamespace()

    dictmodl = build_total_sample_dict(gdat, gmod, 'demo', 0, np.array([10.0, 20.0]))

    assert sorted(dictmodl.keys()) == ['PosteriorSamplesmodl0000', 'PosteriorSamplesmodl0001']
    assert dictmodl['PosteriorSamplesmodl0000']['labl'] == 'Model'
    assert dictmodl['PosteriorSamplesmodl0001']['labl'] is None
    assert dictmodl['PosteriorSamplesmodl0000']['colr'] == 'b'


def test_build_component_sample_dict_creates_component_entries():
    gdat = DummyGdat()
    gdat.numbsampplot = 2
    gdat.booldiag = True
    gdat.fitt = types.SimpleNamespace(
        typemodlenerfitt='sparse',
        listnamecompmodl=['Total', 'Transit', 'StarFlaring'],
    )
    gmod = types.SimpleNamespace(
        listdictsamp=[{
            'Model_Fine_Transit_demo': np.array([[[1.0], [2.0]], [[3.0], [4.0]]]),
            'Model_Fine_StarFlaring_demo': np.array([[[5.0], [6.0]], [[7.0], [8.0]]]),
        }]
    )

    dictmodl = build_component_sample_dict(gdat, gmod, 'demo', 0, np.array([10.0, 20.0]))

    assert sorted(dictmodl.keys()) == [
        'PosteriorSamplesStarFlaring0000',
        'PosteriorSamplesStarFlaring0001',
        'PosteriorSamplesTransit0000',
        'PosteriorSamplesTransit0001',
    ]
    assert dictmodl['PosteriorSamplesTransit0000']['labl'] == 'Transit'
    assert dictmodl['PosteriorSamplesTransit0000']['colr'] == 'r'
    assert dictmodl['PosteriorSamplesStarFlaring0000']['labl'] == 'Flares'
    assert dictmodl['PosteriorSamplesStarFlaring0000']['colr'] == 'g'


def test_retr_pcur_lablpara_for_modes():
    assert retr_pcur_lablpara('0003')[2] == ['Planetary Modulation', 'ppm']
    assert retr_pcur_lablpara('0004')[2] == ['Thermal', 'ppm']


def test_build_pcur_post_for_modes():
    gdat = DummyGdat()
    gdat.numbsamp = 2
    gdat.dictlist = {
        'amplnigh': np.array([[1e-6], [2e-6]]),
        'amplseco': np.array([[3e-6], [4e-6]]),
        'amplplan': np.array([[5e-6], [6e-6]]),
        'amplplanther': np.array([[7e-6], [8e-6]]),
        'amplplanrefl': np.array([[9e-6], [1e-5]]),
        'phasshftplan': np.array([[11.], [12.]]),
        'phasshftplanther': np.array([[13.], [14.]]),
        'phasshftplanrefl': np.array([[15.], [16.]]),
        'albg': np.array([[0.1], [0.2]]),
    }

    listpost3, labl3 = build_pcur_post(gdat, '0003', 0)
    assert labl3[0] == ['Nightside', 'ppm']
    assert np.allclose(listpost3[:, 0], np.array([1., 2.]))
    assert np.allclose(listpost3[:, 2], np.array([5., 6.]))
    assert np.allclose(listpost3[:, 5], np.array([11., 12.]))

    listpost4, labl4 = build_pcur_post(gdat, '0004', 0)
    assert labl4[4] == ['Thermal Phase shift', 'deg']
    assert np.allclose(listpost4[:, 2], np.array([7., 8.]))
    assert np.allclose(listpost4[:, 4], np.array([13., 14.]))


def test_retr_pcur_panel_series_helpers():
    gdat = DummyGdat()
    typemodl = '0003'
    gdat.timeoffs = 100
    gdat.time = [[np.array([100.0, 101.0])]]
    gdat.arrytser = {
        'Detrended' + typemodl: [[np.array([[100.0, 1.1], [101.0, 1.2]])]],
        'modltotl' + typemodl: [[ [np.array([[100.0, 1.3], [101.0, 1.4]])] ]],
    }
    gdat.dicterrr = {'amplnigh': np.array([[0.01]])}
    gmod = types.SimpleNamespace(
        arrypcur={
            'DetrendedQuadratureCentered' + typemodl: [[ [np.array([[0.1, 1.5], [0.2, 1.6]])] ]],
            'DetrendedQuadratureCentered' + typemodl + 'bindtotl': [[ [np.array([[0.1, 1.7, 0.01], [0.2, 1.8, 0.02]])] ]],
            'quadmodl' + typemodl: [[ [np.array([[0.1, 1.9], [0.2, 2.0]])] ]],
        }
    )

    xraw0, yraw0 = retr_pcur_raw_series(gdat, gmod, typemodl, 0, 0, 0, 0)
    assert np.allclose(xraw0, np.array([0.0, 1.0]))
    assert np.allclose(yraw0, np.array([1.11, 1.21]))

    xraw1, yraw1 = retr_pcur_raw_series(gdat, gmod, typemodl, 0, 0, 0, 1)
    assert np.allclose(xraw1, np.array([0.1, 0.2]))
    assert np.allclose(yraw1, np.array([1.51, 1.61]))

    xb0, yb0, eb0 = retr_pcur_binned_series(gdat, gmod, typemodl, 0, 0, 0, 0)
    assert xb0 is None and yb0 is None and eb0 is None

    xb2, yb2, eb2 = retr_pcur_binned_series(gdat, gmod, typemodl, 0, 0, 0, 2)
    assert np.allclose(xb2, np.array([0.1, 0.2]))
    assert np.allclose(yb2, np.array([710000., 810000.]))
    assert np.allclose(eb2, np.array([10000., 20000.]))

    xm0, ym0 = retr_pcur_model_series(gdat, gmod, typemodl, 0, 0, 0, 0)
    assert np.allclose(xm0, np.array([0.0, 1.0]))
    assert np.allclose(ym0, np.array([1.31, 1.41]))

    xm2, ym2 = retr_pcur_model_series(gdat, gmod, typemodl, 0, 0, 0, 2)
    assert np.allclose(xm2, np.array([0.1, 0.2]))
    assert np.allclose(ym2, np.array([910000., 1010000.]))


def test_retr_pcur_component_overlay():
    typemodl = '0003'
    gmod = types.SimpleNamespace(
        arrypcur={
            'quadmodlstel' + typemodl: [[ [np.array([[0.1, 1.1], [0.2, 1.2]])] ]],
            'quadmodlelli' + typemodl: [[ [np.array([[0.1, 1.01], [0.2, 1.02]])] ]],
            'quadmodlplan' + typemodl: [[ [np.array([[0.1, 1.03], [0.2, 1.04]])] ]],
            'quadmodlnigh' + typemodl: [[ [np.array([[0.1, 1.05], [0.2, 1.06]])] ]],
            'quadmodlpmod' + typemodl: [[ [np.array([[0.1, 1.07], [0.2, 1.08]])] ]],
        }
    )

    dictline = retr_pcur_component_overlay(gmod, typemodl, 0, 0, 0)

    assert sorted(dictline.keys()) == [
        'Ellipsoidal variation',
        'Planetary',
        'Planetary baseline',
        'Planetary modulation',
        'Stellar baseline',
    ]
    assert np.allclose(dictline['Stellar baseline']['ydat'], np.array([100000., 200000.]))
    assert dictline['Stellar baseline']['color'] == 'orange'
    assert np.allclose(dictline['Ellipsoidal variation']['ydat'], np.array([10000., 20000.]))
    assert dictline['Planetary']['color'] == 'g'


def test_retr_pcur_sample_plot_data():
    typemodl = '0003'
    gdat = DummyGdat()
    gdat.dicterrr = {'amplnigh': np.array([[0.01]])}
    gdat.indxsampplot = [4, 7]
    gdat.listarrypcur = {
        'quadmodl' + typemodl: [[[
            np.array([
                [1.90, 2.00],
                [2.10, 2.20],
            ])
        ]]]
    }
    gmod = types.SimpleNamespace(
        arrypcur={
            'DetrendedQuadratureCenteredBinned': [[[
                np.array([[0.1, 1.50, 0.01], [0.2, 1.60, 0.02]])
            ]]],
            'quadmodl' + typemodl: [[[
                np.array([[0.1, 1.9], [0.2, 2.0]])
            ]]],
        }
    )

    dictbinned, listsamp = retr_pcur_sample_plot_data(gdat, gmod, typemodl, 0, 0, 0)

    assert np.allclose(dictbinned['xdat'], np.array([0.1, 0.2]))
    assert np.allclose(dictbinned['ydat'], np.array([510000., 610000.]))
    assert np.allclose(dictbinned['yerr'], np.array([10000., 20000.]))
    assert len(listsamp) == 2
    assert np.allclose(listsamp[0]['xdat'], np.array([0.1, 0.2]))
    assert np.allclose(listsamp[0]['ydat'], np.array([910000., 1010000.]))
    assert np.allclose(listsamp[1]['ydat'], np.array([1110000., 1210000.]))


def test_build_albg_comparison_data():
    gdat = DummyGdat()
    gdat.dictlist = {
        'albg': np.array([[0.2], [0.4], [0.6]]),
        'albginfo': np.array([[0.1], [0.5], [0.7]]),
    }
    calls = []

    def fake_retr_kdegpdfn(samples, bins, width):
        calls.append((samples.copy(), bins.copy(), width))
        return np.full(bins.size - 1, samples.mean() + width)

    dictalbg = build_albg_comparison_data(gdat, fake_retr_kdegpdfn, kdegstdv=0.02, numbbins=5)

    assert np.allclose(dictalbg['binsalbg'], np.array([0.1, 0.25, 0.4, 0.55, 0.7]))
    assert np.allclose(dictalbg['meanalbg'], np.array([0.175, 0.325, 0.475, 0.625]))
    assert np.allclose(dictalbg['pdfnalbg'], np.full(4, 0.42))
    assert np.allclose(dictalbg['pdfnalbginfo'], np.full(4, 0.45333333333333337))
    assert len(calls) == 2
    assert np.allclose(calls[0][0], np.array([0.2, 0.4, 0.6]))
    assert np.allclose(calls[1][0], np.array([0.1, 0.5, 0.7]))


def test_build_psii_summary_data():
    gdat = DummyGdat()
    gdat.dictlist = {
        'tmptequi': np.array([[100.0], [200.0], [300.0], [400.0], [500.0]]),
        'tmptdayy': np.array([[110.0], [210.0], [310.0], [410.0], [510.0]]),
        'tmptnigh': np.array([[90.0], [190.0], [290.0], [390.0], [490.0]]),
    }
    listpsii = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    dictpsii = build_psii_summary_data(gdat, listpsii, numbbins=4)

    assert dictpsii['gmeatmptequi'] == 300.0
    assert dictpsii['gstdtmptequi'] == 136.0
    assert dictpsii['gmeatmptdayy'] == 310.0
    assert dictpsii['gstdtmptdayy'] == 136.0
    assert dictpsii['gmeatmptnigh'] == 290.0
    assert dictpsii['gstdtmptnigh'] == 136.0
    assert dictpsii['gmeapsii'] == 0.3
    assert np.isclose(dictpsii['gstdpsii'], 0.136)
    assert dictpsii['histpsii'].sum() == 5
    assert np.allclose(dictpsii['binspsii'], np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    assert np.allclose(dictpsii['meanpsii'], np.array([0.15, 0.25, 0.35, 0.45]))


def test_build_psii_kdeg_data():
    calls = []

    def fake_retr_kdeg(listpsii, meanpsii, kdegstdv):
        calls.append((listpsii.copy(), meanpsii.copy(), kdegstdv))
        return meanpsii + kdegstdv

    dictkdeg = build_psii_kdeg_data(np.array([0.1, 0.2, 0.3]), np.array([0.15, 0.25]), fake_retr_kdeg)

    assert dictkdeg['kdegstdvpsii'] == 0.01
    assert np.allclose(dictkdeg['kdegpsii'], np.array([0.16, 0.26]))
    assert len(calls) == 1
    assert np.allclose(calls[0][0], np.array([0.1, 0.2, 0.3]))
    assert np.allclose(calls[0][1], np.array([0.15, 0.25]))
    assert calls[0][2] == 0.01


def test_build_spec_data_groups():
    arrydata = np.arange(29 * 8, dtype=float).reshape(29, 8)

    listindv, listgroup = build_spec_data_groups(arrydata)

    assert len(listindv) == 5
    assert len(listgroup) == 2
    assert listindv[0]['color'] == 'k'
    assert listindv[2]['labltemp'] == '$K_s$ (Kovacs\&Kovacs2019)'
    assert listindv[4]['labltemp'] == 'IRAC $\mu$m (Garhart+2019)'
    assert listindv[1]['xdat'] == arrydata[1, 0]
    assert listindv[1]['ydept'] == arrydata[1, 2]
    assert np.isclose(listindv[1]['yflux'], 1e-9 * arrydata[1, 6])
    assert listgroup[0]['color'] == 'r'
    assert listgroup[1]['labltemp'] == 'HST G141 (Evans+2017)'
    assert np.allclose(listgroup[0]['xdat'], arrydata[5:22, 0])
    assert np.allclose(listgroup[1]['ytemp'], arrydata[22:-1, 4])


def test_build_spec_model_data():
    gdat = DummyGdat()
    gdat.cntrwlenband = np.array([0.6, 0.8])
    gdat.thptband = np.array([0.1, 0.9])
    gdat.amplplantheratmo = 2.5e-5
    gdat.wlenvivi = np.array([1.1, 1.2])
    gdat.specvivi = np.array([3.0, 4.0])
    arrydata = np.arange(6 * 8, dtype=float).reshape(6, 8)
    arrymodl = np.arange(3 * 10, dtype=float).reshape(3, 10)

    dictspecmodl = build_spec_model_data(gdat, arrydata, arrymodl)

    assert np.allclose(dictspecmodl['host']['xdat'], arrymodl[:, 0])
    assert np.allclose(dictspecmodl['host']['ydat'], 1e-9 * arrymodl[:, 9])
    assert np.allclose(dictspecmodl['host']['xthpt'], np.array([0.6, 0.8]))
    assert dictspecmodl['depth']['xavg'] == arrydata[0, 0]
    assert np.isclose(dictspecmodl['depth']['yavg'], 25.0)
    assert np.allclose(dictspecmodl['depth']['yretr'], arrymodl[:, 1])
    assert np.allclose(dictspecmodl['depth']['ygcm'], np.array([3e6, 4e6]))
    assert np.allclose(dictspecmodl['flux']['yretr'], 1e-9 * arrymodl[:, 5])
    assert np.allclose(dictspecmodl['flux']['ybboduppr'], 1e-9 * arrymodl[:, 8])


def test_build_abundance_component_specs():
    listspecabnd = build_abundance_component_specs()

    assert len(listspecabnd) == 17
    assert listspecabnd[0]['file'] == 'CH4.txt'
    assert listspecabnd[0]['label'] == 'CH$_4$'
    assert np.isclose(listspecabnd[0]['xpos'], 10**-12.8)
    assert listspecabnd[6]['file'] == 'H2O.txt'
    assert listspecabnd[6]['label'] == 'H$_2$O'
    assert listspecabnd[11]['file'] == 'NH3.txt'
    assert listspecabnd[11]['label'] == 'NH$_3$'
    assert listspecabnd[-1]['file'] == 'e_.txt'
    assert listspecabnd[-1]['label'] == 'e$^-$'


def test_build_ptem_plot_data():
    dataptem = np.array([
        [1.0, 10.0, 100.0],
        [2.0, 20.0, 200.0],
        [3.0, 30.0, 300.0],
        [4.0, 40.0, 400.0],
    ])
    ctrb = np.array([
        [1.0, 10.0, 100.0],
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])
    thptbandctrb = np.array([0.2, 0.8])

    dictptem = build_ptem_plot_data(dataptem, ctrb, thptbandctrb, samplestep=2)

    assert dictptem['numbsamp'] == 3
    assert np.array_equal(dictptem['listindxsampplot'], np.array([0, 2]))
    assert np.allclose(dictptem['presaxis'], np.array([1.0, 10.0, 100.0]))
    assert np.allclose(dictptem['profslowr'], np.percentile(dataptem, 10, axis=0))
    assert np.allclose(dictptem['profsmedi'], np.percentile(dataptem, 50, axis=0))
    assert np.allclose(dictptem['profsuppr'], np.percentile(dataptem, 90, axis=0))
    expected = np.sum(ctrb[1:, :] * thptbandctrb[:, None], axis=0)
    expected *= 1e-12 / np.amax(expected)
    assert np.allclose(dictptem['ctrbtess'], expected)


def test_build_occurrence_highlights_for_post_and_prior():
    gdat = DummyGdat()
    gdat.dictpost = {'radicomp': np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]])}
    gdat.fitt = types.SimpleNamespace(prio=types.SimpleNamespace(meanpara=types.SimpleNamespace(rratcomp=np.array([0.1, 0.2]))))
    gdat.radistar = 10.0
    gdat.stdvrratcompprio = np.array([0.01, 0.02])
    gdat.dictfact = {'rjre': 2.0}
    gdat.listcolrcomp = ['r', 'b']
    gdat.liststrgcomp = ['b', 'c']
    gmod = types.SimpleNamespace(indxcomp=[0, 1])

    listhighlightpost = build_occurrence_highlights(gdat, gmod, 'post')
    assert len(listhighlightpost) == 2
    assert listhighlightpost[0]['xposlowr'] == 2.0
    assert listhighlightpost[0]['xposmedi'] == 1.5
    assert listhighlightpost[0]['xposuppr'] == 4.0
    assert listhighlightpost[1]['labl'] == 'c'
    assert np.isclose(listhighlightpost[1]['texty'], 0.83)

    listhighlightprio = build_occurrence_highlights(gdat, gmod, 'prio')
    assert np.isclose(listhighlightprio[0]['xposlowr'], 1.8)
    assert np.isclose(listhighlightprio[0]['xposmedi'], 1.0)
    assert np.isclose(listhighlightprio[0]['xposuppr'], 2.2)
    assert listhighlightprio[0]['colr'] == 'r'


def test_build_occurrence_rate_data():
    data = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [4.0, 25.0],
    ])
    occulowr = np.array([8.0, 17.0, 20.0])
    occuuppr = np.array([12.0, 24.0, 31.0])

    dictoccu = build_occurrence_rate_data(data, occulowr, occuuppr)

    assert np.allclose(dictoccu['timeoccu'], np.array([1.0, 2.0, 4.0]))
    assert np.allclose(dictoccu['occumean'], np.array([10.0, 20.0, 25.0]))
    assert np.allclose(dictoccu['occuyerr'][0], np.array([2.0, 4.0, 6.0]))
    assert np.allclose(dictoccu['occuyerr'][1], np.array([2.0, 3.0, 5.0]))
    assert np.allclose(dictoccu['xerr'], np.array([0.5, 0.5, 1.0]))


def test_build_period_ratio_highlights():
    gdat = DummyGdat()
    gdat.dicterrr = {'pericomp': np.array([[2.0, 4.0, 8.0]])}
    gdat.listcolrcomp = ['r', 'g', 'b']
    gmod = types.SimpleNamespace(numbcomp=3, indxcomp=[0, 1, 2])

    listhighlight = build_period_ratio_highlights(gdat, gmod)

    assert len(listhighlight) == 3
    assert np.isclose(listhighlight[0]['ratiperi'], 2.0)
    assert listhighlight[0]['colrprim'] == 'r'
    assert listhighlight[0]['colrseco'] == 'g'
    assert np.isclose(listhighlight[-1]['ratiperi'], 2.0)
    assert listhighlight[-1]['colrprim'] == 'g'
    assert listhighlight[-1]['colrseco'] == 'b'


def test_build_period_ratio_resonances():
    listline = build_period_ratio_resonances((0.0, 10.0))

    assert len(listline) == 6
    assert np.isclose(listline[0]['rati'], 2.0)
    assert np.isclose(listline[0]['textx'], 2.05)
    assert np.isclose(listline[0]['texty'], 9.0)
    assert listline[0]['labl'] == '2:1'
    assert np.isclose(listline[-1]['rati'], 2.5)
    assert listline[-1]['labl'] == '5:2'


def test_build_helium_comparison_data():
    gdat = DummyGdat()
    gdat.dicterrr = {
        'radicomp': np.array([[2.0, 3.0]]),
        'masscompused': np.array([[4.0, 5.0]]),
        'tmptplan': np.array([[600.0, 700.0]]),
        'duratrantotl': np.array([[2.0, 4.0]]),
    }
    gdat.radistar = 2.0
    gdat.jmagsyst = 10.4
    gdat.dictfact = {'rjre': 2.0}
    calls = []

    def fake_retr_scalheig(tmptplan, masscomp, radicomp):
        calls.append((tmptplan, masscomp, radicomp))
        return np.asarray(radicomp) * 0.5

    dictheli = build_helium_comparison_data(gdat, fake_retr_scalheig)

    assert len(dictheli['listscenario']) == 2
    assert dictheli['listscenario'][0]['name'] == 'wasp107'
    assert dictheli['listscenario'][1]['name'] == 'target'
    assert np.isclose(dictheli['listscenario'][0]['radicomp'], 1.848)
    assert np.allclose(dictheli['listscenario'][1]['scalheig'], np.array([1.0, 1.5]))
    assert len(dictheli['listscenario'][1]['listtran']) == 5
    assert dictheli['listscenario'][1]['listtran'][0]['numbtran'] == 1
    assert np.allclose(dictheli['fact'], np.array([2e6, 4.5e6]))
    assert np.allclose(dictheli['factstdv'], np.sqrt(10 ** ((-9.4 + 10.4) / 2.5) * 2.74 / np.array([2.0, 4.0])))
    assert len(calls) == 2


def test_build_magnitude_population_plot_data():
    gdat = DummyGdat()
    gdat.vmagsyst = 10.0
    gdat.jmagsyst = 9.0
    gdat.massstar = 8.0
    dictpopl = {
        'vmagsyst': np.array([11.0, 12.0, 13.0, 14.0]),
        'jmagsyst': np.array([8.0, 9.0, 10.0, 11.0]),
        'massstar': np.array([1.0, 8.0, 27.0, 64.0]),
        'numbplanstar': np.array([2, 4, 5, 6]),
        'numbplantranstar': np.array([1, 5, 2, 6]),
        'namestar': np.array(['A', 'B', 'C', 'D']),
    }
    gmod = types.SimpleNamespace(numbcomp=3)

    dictmagt0 = build_magnitude_population_plot_data(gdat, gmod, dictpopl, 0, 0)
    assert dictmagt0['strgvarbmagt'] == 'vmag'
    assert dictmagt0['lablxaxi'] == 'V Magnitude'
    assert np.array_equal(dictmagt0['indx'], np.array([1, 2, 3]))
    assert np.allclose(dictmagt0['varbnorm'], np.array([12.0, 13.0, 14.0]))
    assert dictmagt0['listlabel'][0]['name'] == 'B'

    dictmagt3 = build_magnitude_population_plot_data(gdat, gmod, dictpopl, 3, 1)
    assert dictmagt3['strgvarbmagt'] == 'rvelsemascal_jmag'
    assert dictmagt3['lablxaxi'] == '$K^{\prime}_{J}$'
    assert np.array_equal(dictmagt3['indx'], np.array([1, 3]))
    assert np.isclose(dictmagt3['varbtargnorm'], 0.6309573444801932)
    assert dictmagt3['listlabel'][0]['name'] == 'B'
    assert dictmagt3['listlabel'][1]['name'] == 'D'


def test_build_population_feature_plot_config():
    dictplotcnfg = build_population_feature_plot_config()

    assert dictplotcnfg['liststrgtext'] == ['notx', 'text']
    assert dictplotcnfg['liststrgfeatpairplot'] == [
        ['radicomp', 'tmptplan'],
        ['radicomp', 'tsmm'],
        ['tmptplan', 'tsmm'],
        ['tmptplan', 'vesc0060'],
    ]
    assert dictplotcnfg['liststrgsort'] == ['none', 'tsmm']
    assert dictplotcnfg['liststrgvarb'][0] == 'pericomp'
    assert dictplotcnfg['liststrgvarb'][-1] == 'tagestar'
    assert dictplotcnfg['liststrgfeatcsvv'] == ['rascstar', 'declstar', 'radicomp', 'masscomp', 'tmptplan', 'jmagsyst', 'radistar', 'tsmm']


def test_build_population_sort_plot_data():
    dicttempmerg = {
        'nameplan': np.array(['a', 'b', 'c', 'd']),
        'tsmm': np.array([2.0, np.nan, 5.0, 3.0]),
        'radicomp': np.array([10.0, 20.0, 30.0, np.nan]),
        'tmptplan': np.array([100.0, 200.0, 300.0, 400.0]),
    }

    dictnone = build_population_sort_plot_data(dicttempmerg, 'none', 'notx', 'radicomp', 'tmptplan', 2)
    assert dictnone['boolmakeplot'] is True
    assert dictnone['indxcompsort'] is None
    assert dictnone['listtext'] == []

    dictskip = build_population_sort_plot_data(dicttempmerg, 'none', 'text', 'radicomp', 'tmptplan', 2)
    assert dictskip['boolmakeplot'] is False

    dictsort = build_population_sort_plot_data(dicttempmerg, 'tsmm', 'text', 'radicomp', 'tmptplan', 2)
    assert dictsort['boolmakeplot'] is True
    assert np.array_equal(dictsort['indxcompsort'], np.array([2, 3, 0]))
    assert len(dictsort['listtext']) == 1
    assert dictsort['listtext'][0]['text'] == 'c'
    assert dictsort['listtext'][0]['xdat'] == 30.0


def test_build_feature_pair_guides():
    dictpopl = {'tmptplan': np.array([100.0, 200.0, 400.0])}

    dictguide_temp = build_feature_pair_guides(dictpopl, 'tmptplan', 'vesc0060')
    assert np.allclose(dictguide_temp['xlim'], np.array([50.0, 800.0]))
    assert len(dictguide_temp['curves']) == 6
    assert dictguide_temp['labels'] == []
    assert np.isclose(dictguide_temp['curves'][0]['ydat'][0], (50.0 / 40.0) ** 0.5)

    dictguide_mass = build_feature_pair_guides(dictpopl, 'radicomp', 'masscomp')
    assert dictguide_mass['xlim'] is None
    assert len(dictguide_mass['curves']) == 3
    assert len(dictguide_mass['labels']) == 3
    assert dictguide_mass['labels'][0]['text'] == 'Earth-like'
    assert np.isclose(dictguide_mass['curves'][1]['ydat'][0], (0.5 / 0.1813) ** (1.0 / 3.0))

    dictguide_none = build_feature_pair_guides(dictpopl, 'radicomp', 'tmptplan')
    assert dictguide_none == {'xlim': None, 'curves': [], 'labels': []}


def test_build_feature_pair_target_render_plan():
    gdat = DummyGdat()
    gdat.dicterrr = {
        'radicomp': np.array([[1.0, 2.0], [0.1, 0.2], [0.3, 0.4]]),
        'tmptplan': np.array([[100.0, 200.0], [10.0, 20.0], [30.0, 40.0]]),
        'massstar': np.array([[1.5, 1.6], [0.01, 0.02], [0.03, 0.04]]),
    }
    gdat.listfeatstar = ['massstar']
    gdat.listcolrcomp = ['r', 'b']
    gdat.liststrgcomp = ['b', 'c']
    gdat.labltarg = 'Target'
    gmod = types.SimpleNamespace(indxcomp=[0, 1])

    dictplan_comp = build_feature_pair_target_render_plan(gdat, gmod, 'radicomp', 'tmptplan')
    assert len(dictplan_comp['draw']) == 2
    assert dictplan_comp['draw'][0]['kind'] == 'errorbar'
    assert dictplan_comp['draw'][0]['color'] == 'r'
    assert len(dictplan_comp['text']) == 2
    assert dictplan_comp['text'][0]['text'] == r'\textbf{b}'

    dictplan_star = build_feature_pair_target_render_plan(gdat, gmod, 'foo', 'massstar')
    assert len(dictplan_star['draw']) == 1
    assert dictplan_star['draw'][0]['kind'] == 'axhline'
    assert dictplan_star['draw'][0]['color'] == 'k'
    assert dictplan_star['text'][0]['text'] == 'Target'

    dictplan_bothstar = build_feature_pair_target_render_plan(gdat, gmod, 'massstar', 'massstar')
    assert len(dictplan_bothstar['draw']) == 1
    assert dictplan_bothstar['draw'][0]['kind'] == 'errorbar'
    assert dictplan_bothstar['draw'][0]['color'] == 'k'
    assert dictplan_bothstar['text'][0]['text'] == 'Target'


def test_build_feature_pair_panel_meta():
    gdat = DummyGdat()
    gdat.strgtarg = 'WASP-12'
    gdat.typefileplot = 'png'

    dictpanel = build_feature_pair_panel_meta(
        '/tmp/visu/',
        'radicomp',
        'tmptplan',
        'Radius',
        'Temperature',
        'self',
        'logt',
        gdat,
        'exar',
        'Total',
        'text',
        'tsmm',
        'post',
    )

    assert dictpanel['lablxaxi'] == 'Radius'
    assert dictpanel['lablyaxi'] == 'Temperature'
    assert dictpanel['boolxlog'] is False
    assert dictpanel['boolylog'] is True
    assert dictpanel['path'] == '/tmp/visu/feat_radicomp_tmptplan_WASP-12_exar_Total_text_tsmm_post.png'


def test_build_feature_pair_population_render_plan():
    dicttempmerg = {
        'radicomp': np.array([1.0, 2.0]),
        'tmptplan': np.array([100.0, 200.0]),
    }

    dictpoplplan = build_feature_pair_population_render_plan(dicttempmerg, 'radicomp', 'tmptplan')

    assert dictpoplplan['kind'] == 'errorbar'
    assert np.allclose(dictpoplplan['xdat'], np.array([1.0, 2.0]))
    assert np.allclose(dictpoplplan['ydat'], np.array([100.0, 200.0]))
    assert dictpoplplan['ls'] == ''
    assert dictpoplplan['ms'] == 1
    assert dictpoplplan['marker'] == 'o'
    assert dictpoplplan['color'] == 'k'


def test_build_population_merge_data_and_pair_selection():
    dictpopl = {
        'radicomp': np.array([1.0, 2.0, 3.0]),
        'tmptplan': np.array([100.0, 200.0, 300.0]),
        'nameplan': np.array(['a', 'b', 'c']),
    }
    dicterrr = {
        'radicomp': np.array([[4.0, 5.0]]),
        'tmptplan': np.array([[400.0, 500.0]]),
        'nameplan': np.array([['d', 'e']], dtype=object),
    }
    indxcompfilt = {'Total': np.array([0, 2])}

    dicttempmerg = build_population_merge_data(dictpopl, dicterrr, ['radicomp', 'tmptplan'], indxcompfilt, 'Total')

    assert np.allclose(dicttempmerg['radicomp'], np.array([1.0, 3.0, 4.0, 5.0]))
    assert np.allclose(dicttempmerg['tmptplan'], np.array([100.0, 300.0, 400.0, 500.0]))
    assert list(dicttempmerg['nameplan']) == ['a', 'c', 'd', 'e']
    assert check_feature_pair_selected('radicomp', 'tmptplan', [['radicomp', 'tmptplan']]) is True
    assert check_feature_pair_selected('tmptplan', 'radicomp', [['radicomp', 'tmptplan']]) is False