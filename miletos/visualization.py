import os

import matplotlib.pyplot as plt
import numpy as np


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