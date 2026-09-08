import os

import numpy as np
import pandas as pd


def _is_scalar_output(valu):
    """Return whether a value should be serialized into summary CSV outputs."""

    boolscal = isinstance(valu, str) or isinstance(valu, float) or isinstance(valu, int) or isinstance(valu, bool)

    return boolscal


def write_target_output_csv(gdat, typeverb=1):
    """Write scalar workflow outputs for one target to a CSV file."""

    path = gdat.pathdatatarg + 'miletos_output.csv'
    with open(path, 'w') as objtfile:
        for name, valu in gdat.dictmileoutp.items():
            if _is_scalar_output(valu):
                objtfile.write('%s, ' % name)
            if isinstance(valu, str):
                objtfile.write('%s' % valu)
            elif isinstance(valu, float) or isinstance(valu, int) or isinstance(valu, bool):
                objtfile.write('%g' % valu)
            if _is_scalar_output(valu):
                objtfile.write('\n')
    if typeverb > 0:
        print('Writing to %s...' % path)


def retr_listnamecols(dictmileoutp):
    """Return cluster-output column names after filtering transient entries."""

    listnamecols = []
    for name in dictmileoutp:
        if name.startswith('lygo_pathsaverflx'):
            continue
        if name.startswith('lygo_strgtitlcntpplot'):
            continue
        listnamecols.append(name)

    return listnamecols


def write_cluster_output_csv(gdat, typeverb=1):
    """Write or append scalar workflow outputs to the cluster summary CSV."""

    path = gdat.pathdataclus + 'miletos_cluster_output.csv'
    boolappe = True
    if os.path.exists(path):
        print('Reading from %s...' % path)
        dicttemp = pd.read_csv(path).to_dict(orient='list')
        if gdat.strgtarg in dicttemp['strgtarg']:
            boolappe = False
        boolmakehead = False
    else:
        print('Opening %s...' % path)
        boolmakehead = True

    if typeverb > 0:
        if boolmakehead:
            print('Will construct a header...')
        else:
            print('Will not construct a header...')

    if not boolappe:
        return

    if typeverb > 0:
        print('gdat.dictmileoutp')
        for name in gdat.dictmileoutp:
            if 'path' in name:
                print(name)

    if boolmakehead:
        if typeverb > 0:
            print('Constructing the header...')
        listnamecols = retr_listnamecols(gdat.dictmileoutp)
        with open(path, 'w') as objtfile:
            k = 0
            for name in listnamecols:
                valu = gdat.dictmileoutp[name]
                if _is_scalar_output(valu):
                    if k > 0:
                        objtfile.write(',')
                    objtfile.write('%s' % name)
                    k += 1
                
            objtfile.write('\n')
            _write_cluster_row(objtfile, listnamecols, gdat.dictmileoutp)
    else:
        with open(path, 'r') as objtfile:
            for line in objtfile:
                listnamecols = line.split(',')
                break
        listnamecols[-1] = listnamecols[-1][:-1]

        with open(path, 'a') as objtfile:
            objtfile.write('\n')
            _write_cluster_row(objtfile, listnamecols, gdat.dictmileoutp)

    if typeverb > 0:
        print('Writing to %s...' % path)


def _write_cluster_row(objtfile, listnamecols, dictmileoutp):
    """Write one scalar-output row to an already opened cluster summary CSV."""

    k = 0
    for name in listnamecols:
        valu = dictmileoutp[name]
        if _is_scalar_output(valu):
            if k > 0:
                objtfile.write(',')
            if isinstance(valu, str):
                objtfile.write('%s' % valu)
            else:
                objtfile.write('%g' % valu)
            k += 1


def writ_filealle(gdat, namefile, pathalle, dictalle, dictalledefa, typeverb=1):
    """Write an allesfitter CSV/config file from explicit and default entries."""

    listline = []
    if namefile == 'params.csv':
        listline.append('#name,value,fit,bounds,label,unit\n')

    if dictalle is not None:
        for strg, varb in dictalle.items():
            if namefile == 'params.csv':
                line = strg
                for k, varbtemp in enumerate(varb):
                    if varbtemp is not None:
                        line += ',' + varbtemp
                    else:
                        line += ',' + dictalledefa[strg][k]
                line += '\n'
            else:
                line = strg + ',' + varb + '\n'
            listline.append(line)
    for strg, varb in dictalledefa.items():
        if dictalle is None or strg not in dictalle:
            if namefile == 'params.csv':
                line = strg
                for varbtemp in varb:
                    line += ',' + varbtemp
                line += '\n'
            else:
                line = strg + ',' + varb + '\n'
            listline.append(line)

    pathfile = pathalle + namefile
    with open(pathfile, 'w') as objtfile:
        for line in listline:
            objtfile.write('%s' % line)
    if typeverb > 0:
        print('Writing to %s...' % pathfile)


def write_alle_data_csvs(gdat, gmod, typemodl, typeverb=1):
    """Write allesfitter input time-series CSV files for one model."""

    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            path = gdat.pathalle[typemodl] + gdat.liststrginst[b][p] + '.csv'
            if os.path.exists(path):
                continue

            if gdat.boolinfefoldbind:
                listarrytserbdtrtemp = np.copy(gmod.arrypcur['DetrendedPrimaryCenteredBinned'][b][p][0])
                listarrytserbdtrtemp[:, 0] *= gdat.fitt.prio.meanpara.pericomp[0]
                listarrytserbdtrtemp[:, 0] += gdat.fitt.prio.meanpara.epocmtracomp[0]
            else:
                listarrytserbdtrtemp = gdat.arrytser['Detrended'][b][p]

            if typeverb > 0:
                print('Writing to %s...' % path)
            np.savetxt(path, listarrytserbdtrtemp, delimiter=',', header=gdat.strgheadtser[b])


def write_alle_params_star(gdat, typemodl, typeverb=1):
    """Write the allesfitter stellar-parameter CSV if it does not exist."""

    pathparastar = gdat.pathalle[typemodl] + 'params_star.csv'
    if os.path.exists(pathparastar):
        return pathparastar

    with open(pathparastar, 'w') as objtfile:
        objtfile.write('#R_star,R_star_lerr,R_star_uerr,M_star,M_star_lerr,M_star_uerr,Teff_star,Teff_star_lerr,Teff_star_uerr\n')
        objtfile.write('#R_sun,R_sun,R_sun,M_sun,M_sun,M_sun,K,K,K\n')
        objtfile.write('%g,%g,%g,%g,%g,%g,%g,%g,%g' % (
            gdat.radistar,
            gdat.stdvradistar,
            gdat.stdvradistar,
            gdat.massstar,
            gdat.stdvmassstar,
            gdat.stdvmassstar,
            gdat.tmptstar,
            gdat.stdvtmptstar,
            gdat.stdvtmptstar,
        ))
    if typeverb > 0:
        print('Writing to %s...' % pathparastar)

    return pathparastar


def build_alle_settings_defaults(gdat, gmod, typemodl):
    """Build the default allesfitter settings dictionary for one model."""

    dictallesettdefa = dict()
    if typemodl == 'pfss':
        for j in gmod.indxcomp:
            dictallesettdefa['%s_flux_weighted_PFS' % gdat.liststrgcomp[j]] = 'True'

    dictallesettdefa['fast_fit_width'] = '%.3g' % (np.amax(gdat.fitt.duramask) / 24.)
    dictallesettdefa['multiprocess'] = 'True'
    dictallesettdefa['multiprocess_cores'] = 'all'
    dictallesettdefa['mcmc_nwalkers'] = '100'
    dictallesettdefa['mcmc_total_steps'] = '100'
    dictallesettdefa['mcmc_burn_steps'] = '10'
    dictallesettdefa['mcmc_thin_by'] = '5'

    for p in gdat.indxinst[0]:
        dictallesettdefa['inst_phot'] = '%s' % gdat.liststrginst[0][p]

    for b in gdat.indxdatatser:
        if b == 0:
            strg = 'phot'
        if b == 1:
            strg = 'rv'
        for p in gdat.indxinst[b]:
            dictallesettdefa['inst_%s' % strg] = '%s' % gdat.liststrginst[b][p]
            dictallesettdefa['host_ld_law_%s' % gdat.liststrginst[b][p]] = 'quad'
            dictallesettdefa['host_grid_%s' % gdat.liststrginst[b][p]] = 'very_sparse'
            dictallesettdefa['baseline_flux_%s' % gdat.liststrginst[b][p]] = 'sample_offset'
            if b == 1:
                dictallesettdefa['ln_jitter_rv_%s' % gdat.liststrginst[b][p]] = [
                    '-10',
                    '1',
                    'uniform -20 20',
                    '$\ln{\sigma_{\mathrm{RV;%s}}}$' % gdat.listlablinst[b][p],
                    '',
                ]

    if typemodl == '0003' or typemodl == '0004':
        dictallesettdefa['phase_curve'] = 'True'
        dictallesettdefa['phase_curve_style'] = 'sine_physical'

    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            for j in gmod.indxcomp:
                dictallesettdefa['%s_grid_%s' % (gdat.liststrgcomp[j], gdat.liststrginst[b][p])] = 'very_sparse'

        if gdat.numbinst[b] > 0:
            if b == 0:
                strg = 'companions_phot'
            if b == 1:
                strg = 'companions_rv'
            varb = ''
            cntr = 0
            for j in gmod.indxcomp:
                if cntr != 0:
                    varb += ' '
                varb += '%s' % gdat.liststrgcomp[j]
                cntr += 1
            dictallesettdefa[strg] = varb

    dictallesettdefa['fast_fit'] = 'True'

    return dictallesettdefa


def write_alle_settings(gdat, gmod, typemodl, writ_filealle_func, typeverb=1):
    """Write the allesfitter settings CSV if it does not exist."""

    pathsett = gdat.pathalle[typemodl] + 'settings.csv'
    if os.path.exists(pathsett):
        return pathsett

    dictallesettdefa = build_alle_settings_defaults(gdat, gmod, typemodl)
    writ_filealle_func(
        gdat,
        'settings.csv',
        gdat.pathalle[typemodl],
        gdat.dictdictallesett[typemodl],
        dictallesettdefa,
        typeverb=typeverb,
    )

    return pathsett


def build_alle_params_defaults(gdat, gmod, typemodl):
    """Build the effective default params.csv dictionary for one model."""

    dictalleparadefa = dict()
    for j in gmod.indxcomp:
        strgrrat = '%s_rr' % gdat.liststrgcomp[j]
        strgrsma = '%s_rsuma' % gdat.liststrgcomp[j]
        strgcosi = '%s_cosi' % gdat.liststrgcomp[j]
        strgepoc = '%s_epoch' % gdat.liststrgcomp[j]
        strgperi = '%s_period' % gdat.liststrgcomp[j]
        strgecos = '%s_f_c' % gdat.liststrgcomp[j]
        strgesin = '%s_f_s' % gdat.liststrgcomp[j]
        strgrvelsema = '%s_K' % gdat.liststrgcomp[j]

        dictalleparadefa[strgrrat] = [
            '%f' % gdat.fitt.prio.meanpara.rratcomp[j],
            '1',
            'uniform 0 %f' % (4 * gdat.fitt.prio.meanpara.rratcomp[j]),
            '$R_{%s} / R_\star$' % gdat.liststrgcomp[j],
            '',
        ]
        dictalleparadefa[strgrsma] = [
            '%f' % gdat.fitt.prio.meanpara.rsmacomp[j],
            '1',
            'uniform 0 %f' % (4 * gdat.fitt.prio.meanpara.rsmacomp[j]),
            '$(R_\star + R_{%s}) / a_{%s}$' % (gdat.liststrgcomp[j], gdat.liststrgcomp[j]),
            '',
        ]
        dictalleparadefa[strgcosi] = [
            '%f' % gdat.fitt.prio.meanpara.cosicomp[j],
            '1',
            'uniform 0 %f' % max(0.1, 4 * gdat.fitt.prio.meanpara.cosicomp[j]),
            '$\cos{i_{%s}}$' % gdat.liststrgcomp[j],
            '',
        ]
        dictalleparadefa[strgepoc] = [
            '%f' % gdat.fitt.prio.meanpara.epocmtracomp[j],
            '1',
            'uniform %f %f' % (
                gdat.fitt.prio.meanpara.epocmtracomp[j] - gdat.stdvepocmtracompprio[j],
                gdat.fitt.prio.meanpara.epocmtracomp[j] + gdat.stdvepocmtracompprio[j],
            ),
            '$T_{0;%s}$' % gdat.liststrgcomp[j],
            '$\mathrm{BJD}$',
        ]
        dictalleparadefa[strgperi] = [
            '%f' % gdat.fitt.prio.meanpara.pericomp[j],
            '1',
            'uniform %f %f' % (
                gdat.fitt.prio.meanpara.pericomp[j] - 3. * gdat.stdvpericompprio[j],
                gdat.fitt.prio.meanpara.pericomp[j] + 3. * gdat.stdvpericompprio[j],
            ),
            '$P_{%s}$' % gdat.liststrgcomp[j],
            'days',
        ]
        dictalleparadefa[strgecos] = [
            '%f' % gdat.ecoscompprio[j],
            '0',
            'uniform -0.9 0.9',
            '$\sqrt{e_{%s}} \cos{\omega_{%s}}$' % (gdat.liststrgcomp[j], gdat.liststrgcomp[j]),
            '',
        ]
        dictalleparadefa[strgesin] = [
            '%f' % gdat.esincompprio[j],
            '0',
            'uniform -0.9 0.9',
            '$\sqrt{e_{%s}} \sin{\omega_{%s}}$' % (gdat.liststrgcomp[j], gdat.liststrgcomp[j]),
            '',
        ]
        dictalleparadefa[strgrvelsema] = [
            '%f' % gdat.rvelsemaprio[j],
            '0',
            'uniform %f %f' % (
                max(0, gdat.rvelsemaprio[j] - 5 * gdat.stdvrvelsemaprio[j]),
                gdat.rvelsemaprio[j] + 5 * gdat.stdvrvelsemaprio[j],
            ),
            '$K_{%s}$' % gdat.liststrgcomp[j],
            '',
        ]

        if typemodl == '0003' or typemodl == '0004':
            for b in gdat.indxdatatser:
                if b != 0:
                    continue
                for p in gdat.indxinst[b]:
                    strgsbrt = '%s_sbratio_' % gdat.liststrgcomp[j] + gdat.liststrginst[b][p]
                    dictalleparadefa[strgsbrt] = [
                        '1e-3',
                        '1',
                        'uniform 0 1',
                        '$J_{%s; \mathrm{%s}}$' % (gdat.liststrgcomp[j], gdat.listlablinst[b][p]),
                        '',
                    ]
                    dictalleparadefa['%s_phase_curve_beaming_%s' % (gdat.liststrgcomp[j], gdat.liststrginst[b][p])] = [
                        '0', '1', 'uniform 0 10', '$A_\mathrm{beam; %s; %s}$' % (gdat.liststrgcomp[j], gdat.listlablinst[b][p]), ''
                    ]
                    dictalleparadefa['%s_phase_curve_atmospheric_%s' % (gdat.liststrgcomp[j], gdat.liststrginst[b][p])] = [
                        '0', '1', 'uniform 0 10', '$A_\mathrm{atmo; %s; %s}$' % (gdat.liststrgcomp[j], gdat.listlablinst[b][p]), ''
                    ]
                    dictalleparadefa['%s_phase_curve_ellipsoidal_%s' % (gdat.liststrgcomp[j], gdat.liststrginst[b][p])] = [
                        '0', '1', 'uniform 0 10', '$A_\mathrm{elli; %s; %s}$' % (gdat.liststrgcomp[j], gdat.listlablinst[b][p]), ''
                    ]

        if typemodl == '0003':
            for b in gdat.indxdatatser:
                if b != 0:
                    continue
                for p in gdat.indxinst[b]:
                    maxmshft = 0.25 * gdat.fitt.prio.meanpara.pericomp[j]
                    minmshft = -maxmshft
                    dictalleparadefa['%s_phase_curve_atmospheric_shift_%s' % (gdat.liststrgcomp[j], gdat.liststrginst[b][p])] = [
                        '0',
                        '1',
                        'uniform %.3g %.3g' % (minmshft, maxmshft),
                        '$\Delta_\mathrm{%s; %s}$' % (gdat.liststrgcomp[j], gdat.listlablinst[b][p]),
                        '',
                    ]

    for b in gdat.indxdatatser:
        for p in gdat.indxinst[b]:
            strgldc1 = 'host_ldc_q1_%s' % gdat.liststrginst[b][p]
            strgldc2 = 'host_ldc_q2_%s' % gdat.liststrginst[b][p]
            strgscal = 'ln_err_flux_%s' % gdat.liststrginst[b][p]
            strgbaseoffs = 'baseline_offset_flux_%s' % gdat.liststrginst[b][p]
            dictalleparadefa[strgldc1] = ['0.5', '1', 'uniform 0 1', '$q_{1; \mathrm{%s}}$' % gdat.listlablinst[b][p], '']
            dictalleparadefa[strgldc2] = ['0.5', '1', 'uniform 0 1', '$q_{2; \mathrm{%s}}$' % gdat.listlablinst[b][p], '']
            dictalleparadefa[strgscal] = ['-7', '1', 'uniform -10 -4', '$\ln{\sigma_\mathrm{%s}}$' % gdat.listlablinst[b][p], '']
            dictalleparadefa[strgbaseoffs] = ['0', '1', 'uniform -1 1', '$O_{\mathrm{%s}}$' % gdat.listlablinst[b][p], '']
            if b == 1:
                dictalleparadefa['ln_jitter_rv_%s' % gdat.liststrginst[b][p]] = [
                    '-10', '1', 'uniform -20 20', '$\ln{\sigma_{\mathrm{RV;%s}}}$' % gdat.listlablinst[b][p], ''
                ]

    return dictalleparadefa


def write_alle_params(gdat, gmod, typemodl, writ_filealle_func, typeverb=1):
    """Write the allesfitter params.csv file if it does not exist."""

    pathpara = gdat.pathalle[typemodl] + 'params.csv'
    if os.path.exists(pathpara):
        return pathpara

    dictalleparadefa = build_alle_params_defaults(gdat, gmod, typemodl)
    writ_filealle_func(
        gdat,
        'params.csv',
        gdat.pathalle[typemodl],
        gdat.dictdictallepara[typemodl],
        dictalleparadefa,
        typeverb=typeverb,
    )

    return pathpara


def ensure_alle_initial_plot(pathalle, show_initial_guess_func):
    """Ensure the allesfitter initial-guess plot exists."""

    path = pathalle + 'results/initial_guess_b.pdf'
    if not os.path.exists(path):
        show_initial_guess_func(pathalle)

    return path


def ensure_alle_mcmc_run(pathalle, mcmc_fit_func, typeverb=1):
    """Ensure the allesfitter MCMC run artifact exists."""

    path = pathalle + 'results/mcmc_save.h5'
    if not os.path.exists(path):
        mcmc_fit_func(pathalle)
    elif typeverb > 0:
        print('%s exists... Skipping the orbit run.' % path)

    return path


def ensure_alle_final_plots(pathalle, mcmc_output_func):
    """Ensure the allesfitter final plot artifact exists."""

    path = pathalle + 'results/mcmc_corner.pdf'
    if not os.path.exists(path):
        mcmc_output_func(pathalle)

    return path


def load_alle_object(gdat, typemodl, allesclass_func, typeverb=1):
    """Load the allesfitter posterior object for one model into gdat."""

    if typeverb > 0:
        print('Reading from %s...' % gdat.pathalle[typemodl])
    gdat.objtalle[typemodl] = allesclass_func(gdat.pathalle[typemodl])

    return gdat.objtalle[typemodl]


def setp_alle_sampling_meta(gdat, typemodl, settings):
    """Populate allesfitter sampling metadata and sampled-index selection on gdat."""

    gdat.numbsampalle = settings['mcmc_total_steps']
    gdat.numbwalkalle = settings['mcmc_nwalkers']
    gdat.numbsampalleburn = settings['mcmc_burn_steps']
    gdat.numbsampallethin = settings['mcmc_thin_by']

    if type(gdat.objtalle[typemodl].posterior_params) is dict:
        namepara = list(gdat.objtalle[typemodl].posterior_params.keys())[0]
        gdat.numbsamp = gdat.objtalle[typemodl].posterior_params[namepara].size
    else:
        raise ValueError('allesfitter posterior_params must be a dictionary.')

    if gdat.numbsamp > 10000:
        gdat.indxsamp = np.random.choice(np.arange(gdat.numbsamp), size=10000, replace=False)
        gdat.numbsamp = 10000
    else:
        gdat.indxsamp = np.arange(gdat.numbsamp)


def reset_alle_phase_curve_median(objtalle, typemodl, zero_sbratio=False, zero_beaming=False,
                                  zero_ellipsoidal=False, zero_atmospheric=True):
    """Zero selected allesfitter median parameters for component-isolation post-processing."""

    if zero_sbratio:
        objtalle.posterior_params_median['b_sbratio_TESS'] = 0
    if zero_beaming:
        objtalle.posterior_params_median['b_phase_curve_beaming_TESS'] = 0
    if zero_ellipsoidal:
        objtalle.posterior_params_median['b_phase_curve_ellipsoidal_TESS'] = 0
    if zero_atmospheric:
        if typemodl == '0003':
            objtalle.posterior_params_median['b_phase_curve_atmospheric_TESS'] = 0
        else:
            objtalle.posterior_params_median['b_phase_curve_atmospheric_thermal_TESS'] = 0
            objtalle.posterior_params_median['b_phase_curve_atmospheric_reflected_TESS'] = 0

    return objtalle


def load_alle_variant(gdat, typemodl, allesclass_func, zero_sbratio=False, zero_beaming=False,
                      zero_ellipsoidal=False, zero_atmospheric=True):
    """Load an allesfitter object and apply a standard median-parameter reset pattern."""

    gdat.objtalle[typemodl] = allesclass_func(gdat.pathalle[typemodl])
    reset_alle_phase_curve_median(
        gdat.objtalle[typemodl],
        typemodl,
        zero_sbratio=zero_sbratio,
        zero_beaming=zero_beaming,
        zero_ellipsoidal=zero_ellipsoidal,
        zero_atmospheric=zero_atmospheric,
    )

    return gdat.objtalle[typemodl]


def setp_alle_base_detrended(gdat, typemodl, b, p):
    """Populate allesfitter baseline-model and detrended series for one instrument/chunk set."""

    gdat.arrytser['modlbase' + typemodl][b][p] = np.copy(gdat.arrytser['Detrended'][b][p])
    gdat.arrytser['modlbase' + typemodl][b][p][:, 1] = gdat.objtalle[typemodl].get_posterior_median_baseline(
        gdat.liststrginst[b][p], 'flux', xx=gdat.time[b][p]
    )

    gdat.arrytser['Detrended' + typemodl][b][p] = np.copy(gdat.arrytser['Detrended'][b][p])
    gdat.arrytser['Detrended' + typemodl][b][p][:, 1] = (
        gdat.arrytser['Detrended'][b][p][:, 1] - gdat.arrytser['modlbase' + typemodl][b][p][:, 1]
    )

    for y in gdat.indxchun[b][p]:
        gdat.listarrytser['modlbase' + typemodl][b][p][y] = np.copy(gdat.listarrytser['Detrended'][b][p][y])
        gdat.listarrytser['modlbase' + typemodl][b][p][y][:, 1] = gdat.objtalle[typemodl].get_posterior_median_baseline(
            gdat.liststrginst[b][p], 'flux', xx=gdat.listarrytser['modlbase' + typemodl][b][p][y][:, 0]
        )
        gdat.listarrytser['Detrended' + typemodl][b][p][y] = np.copy(gdat.listarrytser['Detrended'][b][p][y])
        gdat.listarrytser['Detrended' + typemodl][b][p][y][:, 1] = (
            gdat.listarrytser['Detrended' + typemodl][b][p][y][:, 1]
            - gdat.listarrytser['modlbase' + typemodl][b][p][y][:, 1]
        )


def write_quad_bindtotl_csv(gdat, gmod, typemodl, strgpcurcomp, b, p, j, typeverb=1):
    """Write one rebinned quad phase-curve CSV product if it does not exist."""

    path = gdat.pathdatatarg + 'arrypcur_quad_%sbindtotl_%s_%s.csv' % (
        strgpcurcomp,
        gdat.liststrgcomp[j],
        gdat.liststrginst[b][p],
    )
    if os.path.exists(path):
        return path

    if typeverb > 0:
        print('Writing to %s...' % path)
    np.savetxt(
        path,
        gmod.arrypcur['quad%s%sbindtotl' % (strgpcurcomp, typemodl)][b][p][j],
        delimiter=',',
        header=gdat.strgheadpser[b],
    )

    return path


def write_post_pcur_table_csv(gdat, gmod, typemodl, typeverb=1):
    """Write the numeric post-phase-curve summary CSV if it does not exist."""

    path = gdat.pathalle[typemodl] + 'post_pcur_%s_tabl.csv' % typemodl
    if os.path.exists(path):
        return path

    with open(path, 'w') as fileoutp:
        for strgfeat in gdat.dictlist:
            if gdat.dictlist[strgfeat].ndim == 2:
                for j in gmod.indxcomp:
                    fileoutp.write(
                        '%s,%s,%g,%g,%g,%g,%g\\\\\n' % (
                            strgfeat,
                            gdat.liststrgcomp[j],
                            gdat.dictlist[strgfeat][0, j],
                            gdat.dictlist[strgfeat][1, j],
                            gdat.dictlist[strgfeat][2, j],
                            gdat.dicterrr[strgfeat][1, j],
                            gdat.dicterrr[strgfeat][2, j],
                        )
                    )
            else:
                fileoutp.write(
                    '%s,,%g,%g,%g,%g,%g\\\\\n' % (
                        strgfeat,
                        gdat.dictlist[strgfeat][0],
                        gdat.dictlist[strgfeat][1],
                        gdat.dictlist[strgfeat][2],
                        gdat.dicterrr[strgfeat][1],
                        gdat.dicterrr[strgfeat][2],
                    )
                )

    if typeverb > 0:
        print('Writing to %s...' % path)

    return path


def write_post_pcur_command_csv(gdat, gmod, typemodl, typeverb=1):
    """Write the TeX-style post-phase-curve summary CSV if it does not exist."""

    path = gdat.pathalle[typemodl] + 'post_pcur_%s_cmnd.csv' % typemodl
    if os.path.exists(path):
        return path

    with open(path, 'w') as fileoutp:
        for strgfeat in gdat.dictlist:
            if gdat.dictlist[strgfeat].ndim == 2:
                for j in gmod.indxcomp:
                    fileoutp.write(
                        '%s,%s,$%.3g \substack{+%.3g \\\\ -%.3g}$\\\\\n' % (
                            strgfeat,
                            gdat.liststrgcomp[j],
                            gdat.dicterrr[strgfeat][0, j],
                            gdat.dicterrr[strgfeat][1, j],
                            gdat.dicterrr[strgfeat][2, j],
                        )
                    )
            else:
                fileoutp.write(
                    '%s,,$%.3g \substack{+%.3g \\\\ -%.3g}$\\\\\n' % (
                        strgfeat,
                        gdat.dicterrr[strgfeat][0],
                        gdat.dicterrr[strgfeat][1],
                        gdat.dicterrr[strgfeat][2],
                    )
                )

    if typeverb > 0:
        print('Writing to %s...' % path)

    return path


def write_population_rank_csv(path, dicttempmerg, indxcompsort, liststrgfeatcsvv, liststrgvarb,
                              listlablvarbtotl, typeverb=1):
    """Write one ranked population CSV from merged feature arrays and a sort order."""

    with open(path, 'w') as objtfile:
        strghead = '%4s, %20s' % ('Rank', 'Name')
        for strgfeatcsvv in liststrgfeatcsvv:
            strghead += ', %12s' % listlablvarbtotl[liststrgvarb.index(strgfeatcsvv)]
        strghead += '\n'

        objtfile.write(strghead)
        cntr = 1
        for l in indxcompsort:
            strgline = '%4d, %20s' % (cntr, dicttempmerg['nameplan'][l])
            for strgfeatcsvv in liststrgfeatcsvv:
                strgline += ', %12.4g' % dicttempmerg[strgfeatcsvv][l]
            strgline += '\n'

            objtfile.write(strgline)
            cntr += 1

    if typeverb > 0:
        print('Writing to %s...' % path)

    return path