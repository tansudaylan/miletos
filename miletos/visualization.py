import os

import matplotlib.pyplot as plt


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


def plot_binned_rms(gdat, path, delt, stdvresi):
    """Write the workflow-standard binned-RMS diagnostic plot."""

    if os.path.exists(path):
        return path

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