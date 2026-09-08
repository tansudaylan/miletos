import os

import matplotlib.pyplot as plt
import numpy as np


def retr_lablinst_part(gdat, b, p):
    """Return the instrument-label suffix for summary plot titles."""

    if p is not None and gdat.listlablinst[b][p] != '':
        strglablinst = ', %s' % gdat.listlablinst[b][p]
    else:
        strglablinst = ''

    return strglablinst


def retr_lablcnfg_part(gdat):
    """Return the configuration-label suffix for summary plot titles."""

    if gdat.lablcnfg != '':
        lablcnfgtemp = ', %s' % gdat.lablcnfg
    else:
        lablcnfgtemp = ''

    return lablcnfgtemp


def retr_summary_title(gdat, b, p, e):
    """Return the standard summary-plot title for one target/instrument/band."""

    strglablinst = retr_lablinst_part(gdat, b, p)
    lablcnfgtemp = retr_lablcnfg_part(gdat)

    if e == 0 and gdat.numbener[p] == 1:
        strgtitl = '%s%s%s' % (gdat.labltarg, strglablinst, lablcnfgtemp)
    elif e == 0 and gdat.numbener[p] > 1:
        strgtitl = '%s%s%s, white' % (gdat.labltarg, strglablinst, lablcnfgtemp)
    else:
        strgtitl = '%s%s%s, %g micron' % (
            gdat.labltarg,
            strglablinst,
            lablcnfgtemp,
            gdat.listener[p][e - 1],
        )

    return strgtitl


def retr_summary_extn(prefix, gdat, h, p):
    """Return the standard summary-plot suffix for one workflow view."""

    strgextn = '%s%s' % (prefix, gdat.strgcnfg)
    if gdat.numbener[p] > 1:
        strgextn += gdat.liststrgdatafittiter[h]

    return strgextn


def plot_work_tser(
    plot_tser_func,
    gdat,
    timedata,
    tserdata,
    strgextn,
    strgtitl,
    dictmodl=None,
    lablyaxi=None,
    booldiag=None,
):
    """Call plot_tser with the workflow-standard argument bundle."""

    pathplot = plot_tser_func(
        gdat.pathvisutarg,
        timedata=timedata,
        tserdata=tserdata,
        timeoffs=gdat.timeoffs,
        strgextn=strgextn,
        strgtitl=strgtitl,
        boolwritover=gdat.boolwritover,
        boolbrekmodl=gdat.boolbrekmodl,
        dictmodl=dictmodl,
        lablyaxi=lablyaxi,
        booldiag=booldiag,
    )

    return pathplot


def retr_compmodl_style(namecompmodl):
    """Return the standard label/color pair for a model component."""

    if namecompmodl == 'Total':
        colr = 'b'
        labl = 'Total Model'
    elif namecompmodl == 'Baseline':
        colr = 'orange'
        labl = 'Baseline'
    elif namecompmodl == 'Transit':
        colr = 'r'
        labl = 'Transit'
    elif namecompmodl == 'StarFlaring':
        colr = 'g'
        labl = 'Flares'
    elif namecompmodl == 'excs':
        colr = 'olive'
        labl = 'Excess'
    else:
        raise ValueError('Unknown model component: %s' % namecompmodl)

    return labl, colr


def setp_dictmodl_sample(dictmodl, namevarb, tser, time, labl, colr, alph, booldiag=False):
    """Insert one sample-model series into the plotting dictionary."""

    if booldiag and len(tser) != len(time):
        raise ValueError('Sample series and time axis must have matching lengths.')

    dictmodl[namevarb] = {
        'tser': tser,
        'time': time,
        'labl': labl,
        'colr': colr,
        'alph': alph,
    }


def retr_resi_series(gdat, gmod, strg, e):
    """Return the residual series for the current inference/output mode."""

    if gdat.typeinfe == 'samp':
        if gdat.fitt.typemodlenerfitt == 'full':
            tserdatatemp = np.median(gdat.dictsamp['resi%s' % strg][:, :, e], 0)
        else:
            tserdatatemp = np.median(gmod.listdictsamp[e]['resi%s' % strg][:, :, 0], 0)
    else:
        if gdat.fitt.typemodlenerfitt == 'full':
            tserdatatemp = gdat.dictmlik['resi%s' % strg][:, e]
        else:
            tserdatatemp = gmod.listdictmlik[e]['resi%s' % strg][:, 0]

    return tserdatatemp


def retr_stdvresi_series(gdat, gmod, strg, e):
    """Return the binned residual scatter series for the current inference/output mode."""

    if gdat.typeinfe == 'samp':
        if gdat.fitt.typemodlenerfitt == 'full':
            stdvresi = np.median(gdat.dictsamp['stdvresi%s' % strg][:, :, e], 0)
        else:
            stdvresi = np.median(gmod.listdictsamp[e]['stdvresi%s' % strg][:, :, 0], 0)
    else:
        if gdat.fitt.typemodlenerfitt == 'full':
            stdvresi = gdat.dictmlik['stdvresi%s' % strg][:, e]
        else:
            stdvresi = gmod.listdictmlik[e]['stdvresi' % strg][:, 0]

    return stdvresi


def retr_modl_fine_series(gdat, gmod, namecompmodl, strg, w, e):
    """Return one fine-grid model series for the requested component/sample."""

    if namecompmodl == 'Total':
        namekey = 'Model_Fine_Total_%s' % strg
    else:
        namekey = 'Model_Fine_%s_%s' % (namecompmodl, strg)

    if gdat.fitt.typemodlenerfitt == 'full':
        tsermodl = gdat.dictsamp[namekey][w, :, e]
    else:
        tsermodl = gmod.listdictsamp[e][namekey][w, :, 0]

    return tsermodl


def build_total_sample_dict(gdat, gmod, strg, e, timefine):
    """Return the standard plotting dictionary for total-model posterior samples."""

    dictmodl = dict()
    for w in range(gdat.numbsampplot):
        namevarbsamp = 'PosteriorSamplesmodl%04d' % w
        tsermodl = retr_modl_fine_series(gdat, gmod, 'Total', strg, w, e)
        labl = 'Model' if w == 0 else None
        setp_dictmodl_sample(
            dictmodl,
            namevarbsamp,
            tsermodl,
            timefine,
            labl,
            'b',
            0.2,
            booldiag=gdat.booldiag,
        )

    return dictmodl


def build_component_sample_dict(gdat, gmod, strg, e, timefine):
    """Return the standard plotting dictionary for component posterior samples."""

    dictmodl = dict()
    for namecompmodl in gdat.fitt.listnamecompmodl:
        if namecompmodl == 'Total':
            continue

        labl, colr = retr_compmodl_style(namecompmodl)
        for w in range(gdat.numbsampplot):
            namevarbsamp = 'PosteriorSamples%s%04d' % (namecompmodl, w)
            tsermodl = retr_modl_fine_series(gdat, gmod, namecompmodl, strg, w, e)
            lablsamp = labl if w == 0 else None
            setp_dictmodl_sample(
                dictmodl,
                namevarbsamp,
                tsermodl,
                timefine,
                lablsamp,
                colr,
                0.6,
                booldiag=gdat.booldiag,
            )

    return dictmodl


def retr_pcur_lablpara(typemodl):
    """Return phase-curve posterior labels for the selected allesfitter mode."""

    if typemodl == '0003':
        listlablpara = [
            ['Nightside', 'ppm'],
            ['Secondary', 'ppm'],
            ['Planetary Modulation', 'ppm'],
            ['Thermal', 'ppm'],
            ['Reflected', 'ppm'],
            ['Phase shift', 'deg'],
            ['Geometric Albedo', ''],
        ]
    else:
        listlablpara = [
            ['Nightside', 'ppm'],
            ['Secondary', 'ppm'],
            ['Thermal', 'ppm'],
            ['Reflected', 'ppm'],
            ['Thermal Phase shift', 'deg'],
            ['Reflected Phase shift', 'deg'],
            ['Geometric Albedo', ''],
        ]

    return listlablpara


def build_pcur_post(gdat, typemodl, j):
    """Return the phase-curve posterior summary array for one component."""

    listlablpara = retr_pcur_lablpara(typemodl)
    numbpara = len(listlablpara)
    listpost = np.empty((gdat.numbsamp, numbpara))
    listpost[:, 0] = gdat.dictlist['amplnigh'][:, j] * 1e6
    listpost[:, 1] = gdat.dictlist['amplseco'][:, j] * 1e6
    if typemodl == '0003':
        listpost[:, 2] = gdat.dictlist['amplplan'][:, j] * 1e6
        listpost[:, 3] = gdat.dictlist['amplplanther'][:, j] * 1e6
        listpost[:, 4] = gdat.dictlist['amplplanrefl'][:, j] * 1e6
        listpost[:, 5] = gdat.dictlist['phasshftplan'][:, j]
        listpost[:, 6] = gdat.dictlist['albg'][:, j]
    else:
        listpost[:, 2] = gdat.dictlist['amplplanther'][:, j] * 1e6
        listpost[:, 3] = gdat.dictlist['amplplanrefl'][:, j] * 1e6
        listpost[:, 4] = gdat.dictlist['phasshftplanther'][:, j]
        listpost[:, 5] = gdat.dictlist['phasshftplanrefl'][:, j]
        listpost[:, 6] = gdat.dictlist['albg'][:, j]

    return listpost, listlablpara


def retr_pcur_raw_series(gdat, gmod, typemodl, b, p, j, k):
    """Return raw-data panel series for a phase-curve plot panel."""

    if k == 0:
        xdat = gdat.time[b][p] - gdat.timeoffs
        ydat = gdat.arrytser['Detrended' + typemodl][b][p][:, 1] + gdat.dicterrr['amplnigh'][0, 0]
    else:
        xdat = gmod.arrypcur['DetrendedQuadratureCentered' + typemodl][b][p][j][:, 0]
        ydat = gmod.arrypcur['DetrendedQuadratureCentered' + typemodl][b][p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0]

    return xdat, ydat


def retr_pcur_binned_series(gdat, gmod, typemodl, b, p, j, k):
    """Return binned-data panel series for a phase-curve plot panel."""

    if k == 0:
        return None, None, None

    xdat = gmod.arrypcur['DetrendedQuadratureCentered' + typemodl + 'bindtotl'][b][p][j][:, 0]
    ydat = gmod.arrypcur['DetrendedQuadratureCentered' + typemodl + 'bindtotl'][b][p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0]
    yerr = np.copy(gmod.arrypcur['DetrendedQuadratureCentered' + typemodl + 'bindtotl'][b][p][j][:, 2])
    if k == 2:
        ydat = (ydat - 1) * 1e6
        yerr = yerr * 1e6

    return xdat, ydat, yerr


def retr_pcur_model_series(gdat, gmod, typemodl, b, p, j, k):
    """Return model panel series for a phase-curve plot panel."""

    if k > 0:
        xdat = gmod.arrypcur['quadmodl' + typemodl][b][p][j][:, 0]
        ydat = gmod.arrypcur['quadmodl' + typemodl][b][p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0]
    else:
        xdat = gdat.arrytser['modltotl' + typemodl][b][p][j][:, 0] - gdat.timeoffs
        ydat = gdat.arrytser['modltotl' + typemodl][b][p][j][:, 1] + gdat.dicterrr['amplnigh'][0, 0]
    if k == 2:
        ydat = (ydat - 1) * 1e6

    return xdat, ydat


def retr_pcur_component_overlay(gmod, typemodl, b, p, j):
    """Return standard third-panel overlay curves for phase-curve component plots."""

    dictline = {
        'Stellar baseline': {
            'xdat': gmod.arrypcur['quadmodlstel' + typemodl][b][p][j][:, 0],
            'ydat': (gmod.arrypcur['quadmodlstel' + typemodl][b][p][j][:, 1] - 1.0) * 1e6,
            'color': 'orange',
            'ls': '--',
        },
        'Ellipsoidal variation': {
            'xdat': gmod.arrypcur['quadmodlelli' + typemodl][b][p][j][:, 0],
            'ydat': (gmod.arrypcur['quadmodlelli' + typemodl][b][p][j][:, 1] - 1.0) * 1e6,
            'color': 'r',
            'ls': '--',
        },
        'Planetary': {
            'xdat': gmod.arrypcur['quadmodlplan' + typemodl][b][p][j][:, 0],
            'ydat': (gmod.arrypcur['quadmodlplan' + typemodl][b][p][j][:, 1] - 1.0) * 1e6,
            'color': 'g',
            'ls': '--',
        },
        'Planetary baseline': {
            'xdat': gmod.arrypcur['quadmodlnigh' + typemodl][b][p][j][:, 0],
            'ydat': (gmod.arrypcur['quadmodlnigh' + typemodl][b][p][j][:, 1] - 1.0) * 1e6,
            'color': 'olive',
            'ls': '--',
        },
        'Planetary modulation': {
            'xdat': gmod.arrypcur['quadmodlpmod' + typemodl][b][p][j][:, 0],
            'ydat': (gmod.arrypcur['quadmodlpmod' + typemodl][b][p][j][:, 1] - 1.0) * 1e6,
            'color': 'm',
            'ls': '--',
        },
    }

    return dictline


def retr_pcur_sample_plot_data(gdat, gmod, typemodl, b, p, j):
    """Return binned data and posterior sample curves for the phase-curve sample plot."""

    arrybind = gmod.arrypcur['DetrendedQuadratureCenteredBinned'][b][p][j]
    dictbinned = {
        'xdat': arrybind[:, 0],
        'ydat': (arrybind[:, 1] + gdat.dicterrr['amplnigh'][0, 0] - 1.0) * 1e6,
        'yerr': 1e6 * arrybind[:, 2],
    }

    listsamp = []
    xdat = gmod.arrypcur['quadmodl' + typemodl][b][p][j][:, 0]
    for ii, _ in enumerate(gdat.indxsampplot):
        listsamp.append({
            'xdat': xdat,
            'ydat': 1e6 * (gdat.listarrypcur['quadmodl' + typemodl][b][p][j][ii, :] + gdat.dicterrr['amplnigh'][0, 0] - 1.0),
        })

    return dictbinned, listsamp


def build_albg_comparison_data(gdat, retr_kdegpdfn_func, kdegstdv=0.02, numbbins=100):
    """Return shared albedo-comparison bins and KDE curves for TESS-only vs TESS+ATMO samples."""

    minmalbg = min(np.amin(gdat.dictlist['albginfo']), np.amin(gdat.dictlist['albg']))
    maxmalbg = max(np.amax(gdat.dictlist['albginfo']), np.amax(gdat.dictlist['albg']))
    binsalbg = np.linspace(minmalbg, maxmalbg, numbbins)
    meanalbg = (binsalbg[1:] + binsalbg[:-1]) / 2.0
    pdfnalbg = retr_kdegpdfn_func(gdat.dictlist['albg'][:, 0], binsalbg, kdegstdv)
    pdfnalbginfo = retr_kdegpdfn_func(gdat.dictlist['albginfo'][:, 0], binsalbg, kdegstdv)

    return {
        'binsalbg': binsalbg,
        'meanalbg': meanalbg,
        'pdfnalbg': pdfnalbg,
        'pdfnalbginfo': pdfnalbginfo,
    }


def build_psii_summary_data(gdat, listpsii, numbbins=1001):
    """Return percentile summaries and histogram support for psi/temperature comparison plots."""

    gmeatmptequi = np.percentile(gdat.dictlist['tmptequi'][:, 0], 50.0)
    gstdtmptequi = (
        np.percentile(gdat.dictlist['tmptequi'][:, 0], 84.0)
        - np.percentile(gdat.dictlist['tmptequi'][:, 0], 16.0)
    ) / 2.0
    gmeatmptdayy = np.percentile(gdat.dictlist['tmptdayy'][:, 0], 50.0)
    gstdtmptdayy = (
        np.percentile(gdat.dictlist['tmptdayy'][:, 0], 84.0)
        - np.percentile(gdat.dictlist['tmptdayy'][:, 0], 16.0)
    ) / 2.0
    gmeatmptnigh = np.percentile(gdat.dictlist['tmptnigh'][:, 0], 50.0)
    gstdtmptnigh = (
        np.percentile(gdat.dictlist['tmptnigh'][:, 0], 84.0)
        - np.percentile(gdat.dictlist['tmptnigh'][:, 0], 16.0)
    ) / 2.0
    gmeapsii = np.percentile(listpsii, 50.0)
    gstdpsii = (np.percentile(listpsii, 84.0) - np.percentile(listpsii, 16.0)) / 2.0

    histpsii, binspsii = np.histogram(listpsii, numbbins)
    meanpsii = (binspsii[1:] + binspsii[:-1]) / 2.0

    return {
        'gmeatmptequi': gmeatmptequi,
        'gstdtmptequi': gstdtmptequi,
        'gmeatmptdayy': gmeatmptdayy,
        'gstdtmptdayy': gstdtmptdayy,
        'gmeatmptnigh': gmeatmptnigh,
        'gstdtmptnigh': gstdtmptnigh,
        'gmeapsii': gmeapsii,
        'gstdpsii': gstdpsii,
        'histpsii': histpsii,
        'binspsii': binspsii,
        'meanpsii': meanpsii,
    }


def build_psii_kdeg_data(listpsii, meanpsii, retr_kdeg_func, kdegstdv=0.01):
    """Return the standard psi KDE curve and bandwidth used for comparison plots."""

    kdegpsii = retr_kdeg_func(listpsii, meanpsii, kdegstdv)

    return {
        'kdegstdvpsii': kdegstdv,
        'kdegpsii': kdegpsii,
    }


def build_spec_data_groups(arrydata):
    """Return grouped observational series metadata for the spectrum comparison figure."""

    listindv = [
        {'indx': 0, 'color': 'k', 'labltemp': 'TESS (This work)'},
        {'indx': 1, 'color': 'm', 'labltemp': r'Z$^\prime$ (Delrez+2016)'},
        {'indx': 2, 'color': 'purple', 'labltemp': r'$K_s$ (Kovacs\&Kovacs2019)'},
        {'indx': 3, 'color': 'olive', 'labltemp': r'IRAC $\mu$m (Garhart+2019)'},
        {'indx': 4, 'color': 'olive', 'labltemp': r'IRAC $\mu$m (Garhart+2019)'},
    ]
    listgroup = [
        {'slce': slice(5, 22), 'color': 'r', 'labltemp': 'HST G102 (Evans+2019)'},
        {'slce': slice(22, -1), 'color': 'g', 'labltemp': 'HST G141 (Evans+2017)'},
    ]

    for valu in listindv:
        indx = valu['indx']
        valu['xdat'] = arrydata[indx, 0]
        valu['xerr'] = arrydata[indx, 1]
        valu['ydept'] = arrydata[indx, 2]
        valu['ydeer'] = arrydata[indx, 3]
        valu['ytemp'] = arrydata[indx, 4]
        valu['yteer'] = arrydata[indx, 5]
        valu['yflux'] = 1e-9 * arrydata[indx, 6]
        valu['yfler'] = 1e-9 * arrydata[indx, 7]

    for valu in listgroup:
        slce = valu['slce']
        valu['xdat'] = arrydata[slce, 0]
        valu['xerr'] = arrydata[slce, 1]
        valu['ydept'] = arrydata[slce, 2]
        valu['ydeer'] = arrydata[slce, 3]
        valu['ytemp'] = arrydata[slce, 4]
        valu['yteer'] = arrydata[slce, 5]
        valu['yflux'] = 1e-9 * arrydata[slce, 6]
        valu['yfler'] = 1e-9 * arrydata[slce, 7]

    return listindv, listgroup


def build_spec_model_data(gdat, arrydata, arrymodl):
    """Return model-side series metadata for the spectrum comparison figure."""

    return {
        'host': {
            'xdat': arrymodl[:, 0],
            'ydat': 1e-9 * arrymodl[:, 9],
            'xthpt': gdat.cntrwlenband,
            'ythpt': gdat.thptband,
        },
        'depth': {
            'xavg': arrydata[0, 0],
            'yavg': 1e6 * gdat.amplplantheratmo,
            'xdat': arrymodl[:, 0],
            'yretr': arrymodl[:, 1],
            'ybbod': arrymodl[:, 2],
            'ybbodlowr': arrymodl[:, 3],
            'ybboduppr': arrymodl[:, 4],
            'xgcm': gdat.wlenvivi,
            'ygcm': gdat.specvivi * 1e6,
        },
        'flux': {
            'xdat': arrymodl[:, 0],
            'yretr': 1e-9 * arrymodl[:, 5],
            'ybbod': 1e-9 * arrymodl[:, 6],
            'ybbodlowr': 1e-9 * arrymodl[:, 7],
            'ybboduppr': 1e-9 * arrymodl[:, 8],
        },
    }


def build_abundance_component_specs():
    """Return deterministic abundance-panel file, label, and annotation metadata."""

    return [
        {'file': 'CH4.txt', 'label': 'CH$_4$', 'xpos': 10**-12.8, 'ypos': 10**-2.3},
        {'file': 'CO.txt', 'label': 'CO', 'xpos': 10**-2.8, 'ypos': 10**-3.5},
        {'file': 'FeH.txt', 'label': 'FeH', 'xpos': 10**-10.8, 'ypos': 10**-3.5},
        {'file': 'H+.txt', 'label': 'H$^+$', 'xpos': 10**-12.8, 'ypos': 10**-4.1},
        {'file': 'H.txt', 'label': 'H', 'xpos': 10**-1.6, 'ypos': 10**-2},
        {'file': 'H2.txt', 'label': 'H$_2$', 'xpos': 10**-1.6, 'ypos': 10**-2.6},
        {'file': 'H2O.txt', 'label': 'H$_2$O', 'xpos': 10**-8.8, 'ypos': 10**-4.1},
        {'file': 'H_.txt', 'label': 'H$^-$', 'xpos': 10**-10.0, 'ypos': 10**0.4},
        {'file': 'He.txt', 'label': 'He', 'xpos': 10**-1.6, 'ypos': 10**-4.1},
        {'file': 'K+.txt', 'label': 'K$^+$', 'xpos': 10**-4.4, 'ypos': 10**-4.8},
        {'file': 'K.txt', 'label': 'K', 'xpos': 10**-8.4, 'ypos': 10**-4.8},
        {'file': 'NH3.txt', 'label': 'NH$_3$', 'xpos': 10**-13.6, 'ypos': 10**-4.1},
        {'file': 'Na+.txt', 'label': 'Na$^+$', 'xpos': 10**-4.4, 'ypos': 10**-3.8},
        {'file': 'Na.txt', 'label': 'Na', 'xpos': 10**-6.0, 'ypos': 10**-3.8},
        {'file': 'TiO.txt', 'label': 'TiO', 'xpos': 10**-7.6, 'ypos': 10**-2},
        {'file': 'VO.txt', 'label': 'VO', 'xpos': 10**-6.0, 'ypos': 10**-2},
        {'file': 'e_.txt', 'label': 'e$^-$', 'xpos': 10**-5.6, 'ypos': 10**-0.8},
    ]


def build_ptem_plot_data(dataptem, ctrb, thptbandctrb, samplestep=100):
    """Return PT percentile curves, sampled posterior rows, and weighted contribution profile."""

    numbsamp = dataptem.shape[0] - 1
    indxsamp = np.arange(numbsamp)
    presaxis = dataptem[0, :]
    listindxsampplot = indxsamp[::samplestep]
    profslowr = np.percentile(dataptem, 10, axis=0)
    profsmedi = np.percentile(dataptem, 50, axis=0)
    profsuppr = np.percentile(dataptem, 90, axis=0)

    ctrbtess = np.sum(ctrb[1:, :] * thptbandctrb[:, None], axis=0)
    ctrbtess *= 1e-12 / np.amax(ctrbtess)

    return {
        'numbsamp': numbsamp,
        'listindxsampplot': listindxsampplot,
        'presaxis': presaxis,
        'profslowr': profslowr,
        'profsmedi': profsmedi,
        'profsuppr': profsuppr,
        'ctrbtess': ctrbtess,
    }


def build_occurrence_highlights(gdat, gmod, strgpdfn):
    """Return target highlight spans and labels for the occurrence-rate panel."""

    listhighlight = []
    for jj, j in enumerate(gmod.indxcomp):
        if strgpdfn == 'post':
            xposlowr = gdat.dictpost['radicomp'][0, j]
            xposmedi = gdat.dictpost['radicomp'][1, j]
            xposuppr = gdat.dictpost['radicomp'][2, j]
        else:
            xposmedi = gdat.fitt.prio.meanpara.rratcomp[j] * gdat.radistar
            xposlowr = xposmedi - gdat.stdvrratcompprio[j] * gdat.radistar
            xposuppr = xposmedi + gdat.stdvrratcompprio[j] * gdat.radistar

        xposlowr *= gdat.dictfact['rjre']
        xposuppr *= gdat.dictfact['rjre']
        listhighlight.append({
            'xposlowr': xposlowr,
            'xposmedi': xposmedi,
            'xposuppr': xposuppr,
            'colr': gdat.listcolrcomp[j],
            'labl': gdat.liststrgcomp[j],
            'textx': 0.7,
            'texty': 0.9 - jj * 0.07,
        })

    return listhighlight


def build_occurrence_rate_data(data, occulowr, occuuppr):
    """Return occurrence-rate support arrays for the population comparison panel."""

    timeoccu = data[:, 0]
    occumean = data[:, 1]
    occuyerr = np.empty((2, occumean.size))
    occuyerr[0, :] = occuuppr - occumean
    occuyerr[1, :] = occumean - occulowr

    xerr = (timeoccu[1:] - timeoccu[:-1]) / 2.0
    xerr = np.concatenate([xerr[0, None], xerr])

    return {
        'timeoccu': timeoccu,
        'occumean': occumean,
        'occuyerr': occuyerr,
        'xerr': xerr,
    }


def build_period_ratio_highlights(gdat, gmod):
    """Return system-specific period-ratio marker metadata for the histogram panel."""

    listhighlight = []
    if gmod.numbcomp <= 1:
        return listhighlight

    for j in gmod.indxcomp:
        for jj in gmod.indxcomp:
            if gdat.dicterrr['pericomp'][0, j] > gdat.dicterrr['pericomp'][0, jj]:
                ratiperi = gdat.dicterrr['pericomp'][0, j] / gdat.dicterrr['pericomp'][0, jj]
                listhighlight.append({
                    'ratiperi': ratiperi,
                    'colrprim': gdat.listcolrcomp[jj],
                    'colrseco': gdat.listcolrcomp[j],
                })

    return listhighlight


def build_period_ratio_resonances(ylim, listreso=None):
    """Return resonance guide-line and label metadata for the period-ratio histogram."""

    if listreso is None:
        listreso = [[2., 1.], [3., 2.], [4., 3.], [5., 4.], [5., 3.], [5., 2.]]

    ydatlabl = 0.9 * ylim[1] + ylim[0]
    listline = []
    for perifrst, periseco in listreso:
        rati = perifrst / periseco
        listline.append({
            'rati': rati,
            'textx': rati + 0.05,
            'texty': ydatlabl,
            'labl': '%d:%d' % (perifrst, periseco),
        })

    return listline


def build_helium_comparison_data(
    gdat,
    retr_scalheig_func,
    stdvnirs=0.24e-2,
    duratranplanwasp0107=2.74,
    jmagsystwasp0107=9.4,
):
    """Return deterministic WASP-107 and target comparison data for the helium depth plot."""

    listscenario = []
    for a in range(2):
        if a == 1:
            radicomp = gdat.dicterrr['radicomp'][0, :]
            masscomp = gdat.dicterrr['masscompused'][0, :]
            tmptplan = gdat.dicterrr['tmptplan'][0, :]
            duratranplan = gdat.dicterrr['duratrantotl'][0, :]
            radistar = gdat.radistar
            jmagsyst = gdat.jmagsyst
            name = 'target'
        else:
            radicomp = 0.924 * gdat.dictfact['rjre']
            masscomp = 0.119
            tmptplan = 736
            radistar = 0.66
            jmagsyst = jmagsystwasp0107
            duratranplan = duratranplanwasp0107
            name = 'wasp107'

        scalheig = retr_scalheig_func(tmptplan, masscomp, radicomp)
        deptscal = 1e3 * 2.0 * radicomp * scalheig / radistar**2
        dept = 80.0 * deptscal
        factstdv = np.sqrt(10 ** ((-jmagsystwasp0107 + jmagsyst) / 2.5) * duratranplanwasp0107 / duratranplan)
        stdvnirsthis = factstdv * stdvnirs
        listtran = []
        for b in np.arange(1, 6):
            stdvnirsscal = stdvnirsthis / np.sqrt(float(b))
            sigm = dept / stdvnirsscal
            listtran.append({
                'numbtran': int(b),
                'stdvnirsscal': stdvnirsscal,
                'sigm': sigm,
            })

        listscenario.append({
            'name': name,
            'radicomp': radicomp,
            'masscomp': masscomp,
            'tmptplan': tmptplan,
            'duratranplan': duratranplan,
            'radistar': radistar,
            'jmagsyst': jmagsyst,
            'jmagsystwasp0107': jmagsystwasp0107,
            'scalheig': scalheig,
            'deptscal': deptscal,
            'dept': dept,
            'factstdv': factstdv,
            'stdvnirsthis': stdvnirsthis,
            'listtran': listtran,
        })

    fact = listscenario[-1]['deptscal'] / 500e-6

    return {
        'listscenario': listscenario,
        'fact': fact,
        'factstdv': listscenario[-1]['factstdv'],
    }


def build_magnitude_population_plot_data(gdat, gmod, dictpopl, b, a):
    """Return support data for magnitude-versus-planet-count scatter and histogram panels."""

    if b == 0:
        strgvarbmagt = 'vmag'
        lablxaxi = 'V Magnitude'
        varbtarg = gdat.vmagsyst
        varb = dictpopl['vmagsyst']
    elif b == 1:
        strgvarbmagt = 'jmag'
        lablxaxi = 'J Magnitude'
        varbtarg = gdat.jmagsyst
        varb = dictpopl['jmagsyst']
    elif b == 2:
        strgvarbmagt = 'rvelsemascal_vmag'
        lablxaxi = r'$K^{\prime}_{V}$'
        varbtarg = np.sqrt(10 ** (-gdat.vmagsyst / 2.5)) / gdat.massstar ** (2.0 / 3.0)
        varb = np.sqrt(10 ** (-dictpopl['vmagsyst'] / 2.5)) / dictpopl['massstar'] ** (2.0 / 3.0)
    elif b == 3:
        strgvarbmagt = 'rvelsemascal_jmag'
        lablxaxi = r'$K^{\prime}_{J}$'
        varbtarg = np.sqrt(10 ** (-gdat.vmagsyst / 2.5)) / gdat.massstar ** (2.0 / 3.0)
        varb = np.sqrt(10 ** (-dictpopl['jmagsyst'] / 2.5)) / dictpopl['massstar'] ** (2.0 / 3.0)
    else:
        raise ValueError('Unknown magnitude-panel index: %d' % b)

    if a == 0:
        indx = np.where((dictpopl['numbplanstar'] > 3))[0]
    elif a == 1:
        indx = np.where((dictpopl['numbplantranstar'] > 3))[0]
    else:
        raise ValueError('Unknown population subset index: %d' % a)

    if b == 2 or b == 3:
        normfact = max(varbtarg, np.nanmax(varb[indx]))
    else:
        normfact = 1.0

    varbtargnorm = varbtarg / normfact
    varbnorm = varb[indx] / normfact
    indxsort = np.argsort(varbnorm)
    if b == 2 or b == 3:
        indxsort = indxsort[::-1]

    listlabel = []
    listnameaddd = []
    cntr = 0
    maxmnumbname = min(5, varbnorm.size)
    while cntr < varbnorm.size and len(listnameaddd) < maxmnumbname:
        k = indxsort[cntr]
        nameadd = dictpopl['namestar'][indx][k]
        if nameadd not in listnameaddd:
            listlabel.append({
                'xdat': varbnorm[k],
                'ydat': dictpopl['numbplanstar'][indx][k] + 0.5,
                'name': nameadd,
            })
            listnameaddd.append(nameadd)
        cntr += 1

    return {
        'strgvarbmagt': strgvarbmagt,
        'lablxaxi': lablxaxi,
        'indx': indx,
        'varbtargnorm': varbtargnorm,
        'varbnorm': varbnorm,
        'listlabel': listlabel,
    }


def build_population_feature_plot_config():
    """Return fixed configuration lists for population feature-pair plots."""

    liststrgtext = ['notx', 'text']
    liststrgfeatpairplot = [
        ['radicomp', 'tmptplan'],
        ['radicomp', 'tsmm'],
        ['tmptplan', 'tsmm'],
        ['tmptplan', 'vesc0060'],
    ]
    liststrgsort = ['none', 'tsmm']
    liststrgvarb = [
        'pericomp', 'inso', 'vesc0060', 'masscomp',
        'metrhzon', 'metrterr', 'metrplan', 'metrunlo', 'metrseti',
        'smax',
        'tmptstar',
        'rascstar', 'declstar',
        'loecstar', 'laecstar',
        'radistar',
        'massstar',
        'metastar',
        'radicomp', 'tmptplan',
        'metrhabi', 'metrplan',
        'lgalstar', 'bgalstar', 'distsyst', 'vmagsyst',
        'tsmm', 'esmm',
        'vsiistar', 'projoblq',
        'jmagsyst',
        'tagestar',
    ]
    liststrgfeatcsvv = [
        'rascstar', 'declstar', 'radicomp', 'masscomp', 'tmptplan', 'jmagsyst', 'radistar', 'tsmm',
    ]

    return {
        'liststrgtext': liststrgtext,
        'liststrgfeatpairplot': liststrgfeatpairplot,
        'liststrgsort': liststrgsort,
        'liststrgvarb': liststrgvarb,
        'liststrgfeatcsvv': liststrgfeatcsvv,
    }


def build_population_sort_plot_data(dicttempmerg, strgsort, strgtext, strgxaxi, strgyaxi, numbcomptext):
    """Return sort indices, text overlays, and plot gating for population feature-pair panels."""

    boolmakeplot = not ((strgsort != 'none' and strgtext != 'text') or (strgsort == 'none' and strgtext == 'text'))
    indxcompsort = None
    listtext = []

    if strgsort != 'none':
        indxgood = np.where(np.isfinite(dicttempmerg[strgsort]))[0]
        indxsort = np.argsort(dicttempmerg[strgsort][indxgood])[::-1]
        indxcompsort = indxgood[indxsort]

        if strgtext == 'text':
            for ll, l in enumerate(indxcompsort):
                if ll >= numbcomptext:
                    break
                xdat = dicttempmerg[strgxaxi][l]
                ydat = dicttempmerg[strgyaxi][l]
                if np.isfinite(xdat) and np.isfinite(ydat):
                    listtext.append({
                        'xdat': xdat,
                        'ydat': ydat,
                        'text': '%s' % dicttempmerg['nameplan'][l],
                    })

    return {
        'boolmakeplot': boolmakeplot,
        'indxcompsort': indxcompsort,
        'listtext': listtext,
    }


def build_feature_pair_guides(dictpopl, strgxaxi, strgyaxi):
    """Return deterministic guide-overlay metadata for selected population feature-pair panels."""

    dictguide = {
        'xlim': None,
        'curves': [],
        'labels': [],
    }

    if strgxaxi == 'tmptplan' and strgyaxi == 'vesc0060':
        xlim = [0.5 * np.nanmin(dictpopl['tmptplan']), 2.0 * np.nanmax(dictpopl['tmptplan'])]
        arrytmptplan = np.linspace(xlim[0], xlim[1], 1000)
        cons = [1.0, 4.0, 16.0, 18.0, 28.0, 44.0]
        for consitemp in cons:
            dictguide['curves'].append({
                'xdat': arrytmptplan,
                'ydat': (arrytmptplan / 40.0 / consitemp) ** 0.5,
            })
        dictguide['xlim'] = xlim

    if strgxaxi == 'radicomp' and strgyaxi == 'masscomp':
        listlabldenscomp = ['Earth-like', 'Pure Water', 'Pure Iron']
        listdenscomp = [1.0, 0.1813, 1.428]
        listposicomp = [[13.0, 2.6], [4.7, 3.5], [13.0, 1.9]]
        masscompdens = np.linspace(0.5, 16.0)
        for denscomp in listdenscomp:
            dictguide['curves'].append({
                'xdat': masscompdens,
                'ydat': (masscompdens / denscomp) ** (1.0 / 3.0),
            })
        for labl, posi in zip(listlabldenscomp, listposicomp):
            dictguide['labels'].append({
                'xdat': posi[0],
                'ydat': posi[1],
                'text': labl,
            })

    return dictguide


def build_feature_pair_target_render_plan(gdat, gmod, strgxaxi, strgyaxi):
    """Return drawing commands for target overlays in one population feature-pair panel."""

    listdraw = []
    listtext = []
    for j in gmod.indxcomp:
        boolxdat = strgxaxi in gdat.dicterrr
        boolydat = strgyaxi in gdat.dicterrr

        if boolxdat:
            xdat = gdat.dicterrr[strgxaxi][0, j, None]
            xerr = gdat.dicterrr[strgxaxi][1:3, j, None]
        else:
            xdat = None
            xerr = None
        if boolydat:
            ydat = gdat.dicterrr[strgyaxi][0, j, None]
            yerr = gdat.dicterrr[strgyaxi][1:3, j, None]
        else:
            ydat = None
            yerr = None

        if strgxaxi in gdat.listfeatstar and strgyaxi in gdat.listfeatstar:
            listdraw.append({
                'kind': 'errorbar',
                'xdat': xdat,
                'ydat': ydat,
                'xerr': xerr,
                'yerr': yerr,
                'color': 'k',
                'ls': '',
                'marker': 'o',
                'ms': 6,
                'lw': 1,
                'zorder': 2,
            })
            listtext.append({
                'xdat': 0.85,
                'ydat': 0.9 - j * 0.08,
                'text': gdat.labltarg,
                'color': 'k',
                'transform': 'axes',
            })
            break

        if not boolxdat and boolydat:
            if strgyaxi in gdat.listfeatstar:
                listdraw.append({
                    'kind': 'axhline',
                    'ydat': ydat,
                    'color': 'k',
                    'lw': 1,
                    'ls': '--',
                    'zorder': 2,
                })
                listtext.append({
                    'xdat': 0.85,
                    'ydat': 0.9 - j * 0.08,
                    'text': gdat.labltarg,
                    'color': 'k',
                    'transform': 'axes',
                })
                break
            listdraw.append({
                'kind': 'axhline',
                'ydat': ydat,
                'color': gdat.listcolrcomp[j],
                'lw': 1,
                'ls': '--',
                'zorder': 2,
            })

        if not boolydat and boolxdat:
            if strgxaxi in gdat.listfeatstar:
                listdraw.append({
                    'kind': 'axvline',
                    'xdat': xdat,
                    'color': 'k',
                    'lw': 1,
                    'ls': '--',
                    'zorder': 2,
                })
                listtext.append({
                    'xdat': 0.85,
                    'ydat': 0.9 - j * 0.08,
                    'text': gdat.labltarg,
                    'color': 'k',
                    'transform': 'axes',
                })
                break
            listdraw.append({
                'kind': 'axvline',
                'xdat': xdat,
                'color': gdat.listcolrcomp[j],
                'lw': 1,
                'ls': '--',
                'zorder': None,
            })

        if boolxdat and boolydat:
            listdraw.append({
                'kind': 'errorbar',
                'xdat': xdat,
                'ydat': ydat,
                'xerr': xerr,
                'yerr': yerr,
                'color': gdat.listcolrcomp[j],
                'ls': '',
                'marker': 'o',
                'ms': 6,
                'lw': 1,
                'zorder': 2,
            })

        if boolxdat or boolydat:
            listtext.append({
                'xdat': 0.85,
                'ydat': 0.9 - j * 0.08,
                'text': r'\textbf{%s}' % gdat.liststrgcomp[j],
                'color': gdat.listcolrcomp[j],
                'transform': 'axes',
            })

    return {
        'draw': listdraw,
        'text': listtext,
    }


def build_feature_pair_panel_meta(
    pathvisufeatplan,
    strgxaxi,
    strgyaxi,
    lablxaxi,
    lablyaxi,
    scalxaxi,
    scalyaxi,
    gdat,
    strgpopl,
    strgcutt,
    strgtext,
    strgsort,
    strgpdfn,
):
    """Return axis-scale flags and output path metadata for one feature-pair panel."""

    return {
        'lablxaxi': lablxaxi,
        'lablyaxi': lablyaxi,
        'boolxlog': scalxaxi == 'logt',
        'boolylog': scalyaxi == 'logt',
        'path': pathvisufeatplan + 'feat_%s_%s_%s_%s_%s_%s_%s_%s.%s' % (
            strgxaxi,
            strgyaxi,
            gdat.strgtarg,
            strgpopl,
            strgcutt,
            strgtext,
            strgsort,
            strgpdfn,
            gdat.typefileplot,
        ),
    }


def build_feature_pair_population_render_plan(dicttempmerg, strgxaxi, strgyaxi):
    """Return drawing metadata for the population background in one feature-pair panel."""

    return {
        'kind': 'errorbar',
        'xdat': dicttempmerg[strgxaxi],
        'ydat': dicttempmerg[strgyaxi],
        'ls': '',
        'ms': 1,
        'marker': 'o',
        'color': 'k',
    }


def build_population_merge_data(dictpopl, dicterrr, liststrgvarb, indxcompfilt, strgcuttmain):
    """Return merged population-plus-target feature arrays for feature-pair plotting."""

    dicttempmerg = dict()
    for strgxaxi in liststrgvarb + ['nameplan']:
        if strgxaxi not in dictpopl or strgxaxi not in dicterrr:
            continue
        dicttempmerg[strgxaxi] = np.concatenate([
            dictpopl[strgxaxi][indxcompfilt[strgcuttmain]],
            dicterrr[strgxaxi][0, :],
        ])

    return dicttempmerg


def check_feature_pair_selected(strgxaxi, strgyaxi, liststrgfeatpairplot):
    """Return whether a feature pair is enabled for plotting."""

    for pair in liststrgfeatpairplot:
        if strgxaxi == pair[0] and strgyaxi == pair[1]:
            return True

    return False


def plot_binned_rms(gdat, path, delt, stdvresi):
    """Write the workflow-standard binned-RMS diagnostic plot."""

    if os.path.exists(path):
        return path

    delt = np.asarray(delt)
    stdvresi = np.asarray(stdvresi)

    figr, axis = plt.subplots(figsize=gdat.figrsizeydob)
    axis.loglog(delt * 24.0, stdvresi * 1e6, ls='', marker='o', ms=1, label='Binned Std. Dev')
    axis.axvline(gdat.cadetimeplot * 24.0, ls='--', label='Sampling rate')
    axis.set_ylabel('RMS [ppm]')
    axis.set_xlabel('Bin width [hour]')
    axis.legend()
    plt.tight_layout()
    if gdat.typeverb > 0:
        print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close(figr)

    return path