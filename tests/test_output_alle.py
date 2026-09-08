import numpy as np
import types

from miletos.output import build_alle_params_defaults, build_alle_settings_defaults, ensure_alle_final_plots, ensure_alle_initial_plot, ensure_alle_mcmc_run, load_alle_object, load_alle_variant, reset_alle_phase_curve_median, setp_alle_base_detrended, setp_alle_sampling_meta, write_alle_data_csvs, write_alle_params, write_alle_params_star, write_alle_settings, write_population_rank_csv, write_post_pcur_command_csv, write_post_pcur_table_csv, write_quad_bindtotl_csv, writ_filealle


class DummyGdat:
    pass


def test_write_alle_data_csvs_writes_detrended_files(tmp_path):
    gdat = DummyGdat()
    gdat.pathalle = {'fitt': str(tmp_path) + '/'}
    gdat.indxdatatser = [0]
    gdat.indxinst = [[0]]
    gdat.liststrginst = [['TESS']]
    gdat.boolinfefoldbind = False
    gdat.arrytser = {'Detrended': [[np.array([[1.0, 2.0], [3.0, 4.0]])]]}
    gdat.strgheadtser = ['time,flux']
    gdat.typeverb = 0
    gmod = types.SimpleNamespace()

    write_alle_data_csvs(gdat, gmod, 'fitt', typeverb=0)

    text = (tmp_path / 'TESS.csv').read_text(encoding='utf-8')
    assert 'time,flux' in text
    assert '1.000000000000000000e+00,2.000000000000000000e+00' in text


def test_write_alle_params_star_writes_expected_file(tmp_path):
    gdat = DummyGdat()
    gdat.pathalle = {'fitt': str(tmp_path) + '/'}
    gdat.radistar = 1.1
    gdat.stdvradistar = 0.1
    gdat.massstar = 1.2
    gdat.stdvmassstar = 0.2
    gdat.tmptstar = 5000.0
    gdat.stdvtmptstar = 50.0

    path = write_alle_params_star(gdat, 'fitt', typeverb=0)

    text = (tmp_path / 'params_star.csv').read_text(encoding='utf-8')
    assert path.endswith('params_star.csv')
    assert '#R_star,R_star_lerr,R_star_uerr' in text
    assert '1.1,0.1,0.1,1.2,0.2,0.2,5000,50,50' in text


def test_build_alle_settings_defaults_builds_expected_keys():
    gdat = DummyGdat()
    gdat.fitt = types.SimpleNamespace(duramask=np.array([12.0, 24.0]))
    gdat.indxinst = [[0], [0]]
    gdat.indxdatatser = [0, 1]
    gdat.liststrginst = [['TESS'], ['PFS']]
    gdat.listlablinst = [['TESS'], ['PFS']]
    gdat.liststrgcomp = ['b', 'c']
    gdat.numbinst = [1, 1]
    gmod = types.SimpleNamespace(indxcomp=[0, 1])

    dictsett = build_alle_settings_defaults(gdat, gmod, '0003')

    assert dictsett['fast_fit_width'] == '1'
    assert dictsett['inst_phot'] == 'TESS'
    assert dictsett['inst_rv'] == 'PFS'
    assert dictsett['phase_curve'] == 'True'
    assert dictsett['phase_curve_style'] == 'sine_physical'
    assert dictsett['companions_phot'] == 'b c'
    assert dictsett['companions_rv'] == 'b c'
    assert dictsett['b_grid_TESS'] == 'very_sparse'
    assert dictsett['c_grid_PFS'] == 'very_sparse'
    assert dictsett['ln_jitter_rv_PFS'][0] == '-10'


def test_write_alle_settings_writes_expected_file(tmp_path):
    gdat = DummyGdat()
    gdat.pathalle = {'fitt': str(tmp_path) + '/'}
    gdat.dictdictallesett = {'fitt': None}
    gdat.fitt = types.SimpleNamespace(duramask=np.array([24.0]))
    gdat.indxinst = [[0]]
    gdat.indxdatatser = [0]
    gdat.liststrginst = [['TESS']]
    gdat.listlablinst = [['TESS']]
    gdat.liststrgcomp = ['b']
    gdat.numbinst = [1]
    gmod = types.SimpleNamespace(indxcomp=[0])

    def fake_writ_filealle(gdat_inpt, namefile, pathalle, dictalle, dictalledefa, typeverb=1):
        pathfile = pathalle + namefile
        with open(pathfile, 'w', encoding='utf-8') as objtfile:
            for key, value in dictalledefa.items():
                objtfile.write(f'{key},{value}\n')

    path = write_alle_settings(gdat, gmod, 'fitt', fake_writ_filealle, typeverb=0)

    text = (tmp_path / 'settings.csv').read_text(encoding='utf-8')
    assert path.endswith('settings.csv')
    assert 'fast_fit_width,1' in text
    assert 'inst_phot,TESS' in text
    assert 'companions_phot,b' in text


def test_build_alle_params_defaults_builds_expected_keys():
    gdat = DummyGdat()
    gdat.fitt = types.SimpleNamespace(
        prio=types.SimpleNamespace(
            meanpara=types.SimpleNamespace(
                rratcomp=np.array([0.1]),
                rsmacomp=np.array([5.0]),
                cosicomp=np.array([0.05]),
                epocmtracomp=np.array([100.0]),
                pericomp=np.array([2.0]),
            )
        )
    )
    gdat.stdvepocmtracompprio = np.array([0.2])
    gdat.stdvpericompprio = np.array([0.3])
    gdat.ecoscompprio = np.array([0.01])
    gdat.esincompprio = np.array([0.02])
    gdat.rvelsemaprio = np.array([10.0])
    gdat.stdvrvelsemaprio = np.array([1.0])
    gdat.indxdatatser = [0, 1]
    gdat.indxinst = [[0], [0]]
    gdat.liststrgcomp = ['b']
    gdat.liststrginst = [['TESS'], ['PFS']]
    gdat.listlablinst = [['TESS'], ['PFS']]
    gmod = types.SimpleNamespace(indxcomp=[0])

    dictpara = build_alle_params_defaults(gdat, gmod, '0003')

    assert dictpara['b_rr'][0] == '0.100000'
    assert dictpara['b_rsuma'][2] == 'uniform 0 20.000000'
    assert dictpara['b_epoch'][4] == '$\mathrm{BJD}$'
    assert dictpara['b_K'][0] == '10.000000'
    assert dictpara['b_sbratio_TESS'][0] == '1e-3'
    assert dictpara['b_phase_curve_atmospheric_shift_TESS'][2] == 'uniform -0.5 0.5'
    assert dictpara['host_ldc_q1_TESS'][0] == '0.5'
    assert dictpara['ln_jitter_rv_PFS'][2] == 'uniform -20 20'


def test_write_alle_params_writes_expected_file(tmp_path):
    gdat = DummyGdat()
    gdat.pathalle = {'fitt': str(tmp_path) + '/'}
    gdat.dictdictallepara = {'fitt': None}
    gdat.fitt = types.SimpleNamespace(
        prio=types.SimpleNamespace(
            meanpara=types.SimpleNamespace(
                rratcomp=np.array([0.1]),
                rsmacomp=np.array([5.0]),
                cosicomp=np.array([0.05]),
                epocmtracomp=np.array([100.0]),
                pericomp=np.array([2.0]),
            )
        )
    )
    gdat.stdvepocmtracompprio = np.array([0.2])
    gdat.stdvpericompprio = np.array([0.3])
    gdat.ecoscompprio = np.array([0.01])
    gdat.esincompprio = np.array([0.02])
    gdat.rvelsemaprio = np.array([10.0])
    gdat.stdvrvelsemaprio = np.array([1.0])
    gdat.indxdatatser = [0]
    gdat.indxinst = [[0]]
    gdat.liststrgcomp = ['b']
    gdat.liststrginst = [['TESS']]
    gdat.listlablinst = [['TESS']]
    gmod = types.SimpleNamespace(indxcomp=[0])

    def fake_writ_filealle(gdat_inpt, namefile, pathalle, dictalle, dictalledefa, typeverb=1):
        pathfile = pathalle + namefile
        with open(pathfile, 'w', encoding='utf-8') as objtfile:
            for key, value in dictalledefa.items():
                objtfile.write(f'{key},{value[0]}\n')

    path = write_alle_params(gdat, gmod, 'fitt', fake_writ_filealle, typeverb=0)

    text = (tmp_path / 'params.csv').read_text(encoding='utf-8')
    assert path.endswith('params.csv')
    assert 'b_rr,0.100000' in text
    assert 'host_ldc_q1_TESS,0.5' in text


def test_writ_filealle_writes_params_with_overrides(tmp_path):
    dictalle = {'foo': ['1', None, 'uniform 0 1', 'label', 'unit']}
    dictalledefa = {'foo': ['2', '1', 'uniform 0 2', 'labeld', 'unitd']}

    writ_filealle(None, 'params.csv', str(tmp_path) + '/', dictalle, dictalledefa, typeverb=0)

    text = (tmp_path / 'params.csv').read_text(encoding='utf-8')
    assert '#name,value,fit,bounds,label,unit' in text
    assert 'foo,1,1,uniform 0 1,label,unit' in text


def test_ensure_alle_initial_plot_runs_callback(tmp_path):
    pathalle = str(tmp_path) + '/'
    (tmp_path / 'results').mkdir()

    def fake_show_initial_guess(pathbase):
        (tmp_path / 'results' / 'initial_guess_b.pdf').write_text(pathbase, encoding='utf-8')

    path = ensure_alle_initial_plot(pathalle, fake_show_initial_guess)

    assert path.endswith('results/initial_guess_b.pdf')
    assert (tmp_path / 'results' / 'initial_guess_b.pdf').exists()


def test_ensure_alle_mcmc_run_runs_callback(tmp_path):
    pathalle = str(tmp_path) + '/'
    (tmp_path / 'results').mkdir()

    def fake_mcmc_fit(pathbase):
        (tmp_path / 'results' / 'mcmc_save.h5').write_text(pathbase, encoding='utf-8')

    path = ensure_alle_mcmc_run(pathalle, fake_mcmc_fit)

    assert path.endswith('results/mcmc_save.h5')
    assert (tmp_path / 'results' / 'mcmc_save.h5').exists()


def test_ensure_alle_final_plots_runs_callback(tmp_path):
    pathalle = str(tmp_path) + '/'
    (tmp_path / 'results').mkdir()

    def fake_mcmc_output(pathbase):
        (tmp_path / 'results' / 'mcmc_corner.pdf').write_text(pathbase, encoding='utf-8')

    path = ensure_alle_final_plots(pathalle, fake_mcmc_output)

    assert path.endswith('results/mcmc_corner.pdf')
    assert (tmp_path / 'results' / 'mcmc_corner.pdf').exists()


def test_load_alle_object_sets_objtalle():
    gdat = DummyGdat()
    gdat.pathalle = {'fitt': '/tmp/allesfit_fitt/'}
    gdat.objtalle = {}

    def fake_allesclass(path):
        return {'path': path}

    objt = load_alle_object(gdat, 'fitt', fake_allesclass, typeverb=0)

    assert objt == {'path': '/tmp/allesfit_fitt/'}
    assert gdat.objtalle['fitt'] == {'path': '/tmp/allesfit_fitt/'}


def test_setp_alle_sampling_meta_sets_counts_and_indices():
    gdat = DummyGdat()
    gdat.objtalle = {'fitt': types.SimpleNamespace(posterior_params={'theta': np.arange(8)})}

    setp_alle_sampling_meta(
        gdat,
        'fitt',
        {
            'mcmc_total_steps': 100,
            'mcmc_nwalkers': 20,
            'mcmc_burn_steps': 10,
            'mcmc_thin_by': 5,
        },
    )

    assert gdat.numbsampalle == 100
    assert gdat.numbwalkalle == 20
    assert gdat.numbsampalleburn == 10
    assert gdat.numbsampallethin == 5
    assert gdat.numbsamp == 8
    assert np.array_equal(gdat.indxsamp, np.arange(8))


def test_reset_alle_phase_curve_median_zeroes_requested_terms():
    objtalle = types.SimpleNamespace(posterior_params_median={
        'b_sbratio_TESS': 1,
        'b_phase_curve_beaming_TESS': 2,
        'b_phase_curve_ellipsoidal_TESS': 3,
        'b_phase_curve_atmospheric_TESS': 4,
        'b_phase_curve_atmospheric_thermal_TESS': 5,
        'b_phase_curve_atmospheric_reflected_TESS': 6,
    })

    reset_alle_phase_curve_median(objtalle, '0003', zero_sbratio=True, zero_beaming=True, zero_ellipsoidal=False)
    assert objtalle.posterior_params_median['b_sbratio_TESS'] == 0
    assert objtalle.posterior_params_median['b_phase_curve_beaming_TESS'] == 0
    assert objtalle.posterior_params_median['b_phase_curve_ellipsoidal_TESS'] == 3
    assert objtalle.posterior_params_median['b_phase_curve_atmospheric_TESS'] == 0

    reset_alle_phase_curve_median(objtalle, '0004', zero_sbratio=False, zero_beaming=False, zero_ellipsoidal=True)
    assert objtalle.posterior_params_median['b_phase_curve_ellipsoidal_TESS'] == 0
    assert objtalle.posterior_params_median['b_phase_curve_atmospheric_thermal_TESS'] == 0
    assert objtalle.posterior_params_median['b_phase_curve_atmospheric_reflected_TESS'] == 0


def test_load_alle_variant_loads_and_resets_selected_terms():
    gdat = DummyGdat()
    gdat.pathalle = {'0003': '/tmp/allesfit_0003/'}
    gdat.objtalle = {}

    def fake_allesclass(path):
        return types.SimpleNamespace(posterior_params_median={
            'b_sbratio_TESS': 1,
            'b_phase_curve_beaming_TESS': 2,
            'b_phase_curve_ellipsoidal_TESS': 3,
            'b_phase_curve_atmospheric_TESS': 4,
        })

    objt = load_alle_variant(gdat, '0003', fake_allesclass, zero_sbratio=True, zero_ellipsoidal=True)

    assert objt.posterior_params_median['b_sbratio_TESS'] == 0
    assert objt.posterior_params_median['b_phase_curve_ellipsoidal_TESS'] == 0
    assert objt.posterior_params_median['b_phase_curve_atmospheric_TESS'] == 0
    assert objt.posterior_params_median['b_phase_curve_beaming_TESS'] == 2


def test_setp_alle_base_detrended_updates_full_and_chunked_series():
    gdat = DummyGdat()
    typemodl = 'fitt'
    gdat.arrytser = {
        'Detrended': [[np.array([[1.0, 10.0], [2.0, 20.0]])]],
        'modlbase' + typemodl: [[None]],
        'Detrended' + typemodl: [[None]],
    }
    gdat.listarrytser = {
        'Detrended': [[[np.array([[1.0, 5.0], [1.5, 7.0]])]]],
        'modlbase' + typemodl: [[[None]]],
        'Detrended' + typemodl: [[[None]]],
    }
    gdat.liststrginst = [['TESS']]
    gdat.time = [[np.array([1.0, 2.0])]]
    gdat.indxchun = [[[0]]]
    gdat.objtalle = {
        typemodl: types.SimpleNamespace(
            get_posterior_median_baseline=lambda inst, flux, xx: np.array(xx) * 2.0
        )
    }

    setp_alle_base_detrended(gdat, typemodl, 0, 0)

    assert np.allclose(gdat.arrytser['modlbase' + typemodl][0][0][:, 1], np.array([2.0, 4.0]))
    assert np.allclose(gdat.arrytser['Detrended' + typemodl][0][0][:, 1], np.array([8.0, 16.0]))
    assert np.allclose(gdat.listarrytser['modlbase' + typemodl][0][0][0][:, 1], np.array([2.0, 3.0]))
    assert np.allclose(gdat.listarrytser['Detrended' + typemodl][0][0][0][:, 1], np.array([3.0, 4.0]))


def test_write_quad_bindtotl_csv_writes_expected_file(tmp_path):
    gdat = DummyGdat()
    gdat.pathdatatarg = str(tmp_path) + '/'
    gdat.liststrgcomp = ['b']
    gdat.liststrginst = [['TESS']]
    gdat.strgheadpser = ['phase,flux']
    gmod = types.SimpleNamespace(
        arrypcur={'quadDetrendedfittbindtotl': [[[np.array([[0.1, 1.0], [0.2, 2.0]])]]]}
    )

    path = write_quad_bindtotl_csv(gdat, gmod, 'fitt', 'Detrended', 0, 0, 0, typeverb=0)

    text = (tmp_path / 'arrypcur_quad_Detrendedbindtotl_b_TESS.csv').read_text(encoding='utf-8')
    assert path.endswith('arrypcur_quad_Detrendedbindtotl_b_TESS.csv')
    assert 'phase,flux' in text
    assert '1.000000000000000056e-01,1.000000000000000000e+00' in text


def test_write_post_pcur_table_csv_writes_expected_file(tmp_path):
    gdat = DummyGdat()
    gdat.pathalle = {'0003': str(tmp_path) + '/'}
    gdat.dictlist = {
        'feat2d': np.array([[1.0], [2.0], [3.0]]),
        'feat1d': np.array([4.0, 5.0, 6.0]),
    }
    gdat.dicterrr = {
        'feat2d': np.array([[0.1], [0.2], [0.3]]),
        'feat1d': np.array([0.4, 0.5, 0.6]),
    }
    gdat.liststrgcomp = ['b']
    gmod = types.SimpleNamespace(indxcomp=[0])

    path = write_post_pcur_table_csv(gdat, gmod, '0003', typeverb=0)

    text = (tmp_path / 'post_pcur_0003_tabl.csv').read_text(encoding='utf-8')
    assert path.endswith('post_pcur_0003_tabl.csv')
    assert 'feat2d,b,1,2,3,0.2,0.3\\' in text
    assert 'feat1d,,4,5,6,0.5,0.6\\' in text


def test_write_post_pcur_command_csv_writes_expected_file(tmp_path):
    gdat = DummyGdat()
    gdat.pathalle = {'0003': str(tmp_path) + '/'}
    gdat.dictlist = {
        'feat2d': np.array([[1.0], [2.0], [3.0]]),
        'feat1d': np.array([4.0, 5.0, 6.0]),
    }
    gdat.dicterrr = {
        'feat2d': np.array([[0.1], [0.2], [0.3]]),
        'feat1d': np.array([0.4, 0.5, 0.6]),
    }
    gdat.liststrgcomp = ['b']
    gmod = types.SimpleNamespace(indxcomp=[0])

    path = write_post_pcur_command_csv(gdat, gmod, '0003', typeverb=0)

    text = (tmp_path / 'post_pcur_0003_cmnd.csv').read_text(encoding='utf-8')
    assert path.endswith('post_pcur_0003_cmnd.csv')
    assert r'feat2d,b,$0.1 \substack{+0.2 \\ -0.3}$\\' in text
    assert r'feat1d,,$0.4 \substack{+0.5 \\ -0.6}$\\' in text


def test_write_population_rank_csv_writes_expected_file(tmp_path):
    path = str(tmp_path / 'rank.csv')
    dicttempmerg = {
        'nameplan': np.array(['planet-b', 'planet-c']),
        'rascstar': np.array([11.0, 22.0]),
        'declstar': np.array([33.0, 44.0]),
        'radicomp': np.array([1.5, 2.5]),
        'masscomp': np.array([5.5, 6.5]),
        'tmptplan': np.array([700.0, 800.0]),
        'jmagsyst': np.array([9.1, 9.2]),
        'radistar': np.array([1.1, 1.2]),
        'tsmm': np.array([100.0, 200.0]),
    }
    indxcompsort = np.array([1, 0])
    liststrgfeatcsvv = ['rascstar', 'declstar', 'radicomp', 'masscomp', 'tmptplan', 'jmagsyst', 'radistar', 'tsmm']
    liststrgvarb = ['foo', 'rascstar', 'declstar', 'radicomp', 'masscomp', 'tmptplan', 'jmagsyst', 'radistar', 'tsmm']
    listlablvarbtotl = ['Foo', 'RA', 'Dec', 'Radius', 'Mass', 'Temp', 'Jmag', 'Rstar', 'TSM']

    pathout = write_population_rank_csv(
        path,
        dicttempmerg,
        indxcompsort,
        liststrgfeatcsvv,
        liststrgvarb,
        listlablvarbtotl,
        typeverb=0,
    )

    text = (tmp_path / 'rank.csv').read_text(encoding='utf-8')
    assert pathout.endswith('rank.csv')
    assert 'Rank,                 Name' in text
    assert 'RA' in text and 'TSM' in text
    assert '   1,             planet-c' in text
    assert ',           22,' in text
    assert ',          200' in text