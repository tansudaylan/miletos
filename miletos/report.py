from tdpy.verbosity import print
import os
import math

import matplotlib.pyplot as plt
import tdpy


def retr_pathpage(gdat, numbpage):
    """Return the path to a DV summary page image."""

    pathpage = gdat.pathvisutarg + 'Summary_Page%d_%s.png' % (numbpage + 1, gdat.strgtarg)

    return pathpage


def make_dvrp_pages(gdat):
    """Build data-validation report pages and return their paths."""

    listpathdvrp = []
    for w in gdat.indxpage:
        pathplot = retr_pathpage(gdat, w)
        listpathdvrp.append(pathplot)

        if os.path.exists(pathplot):
            continue

        listdictdvrp = gdat.listdictdvrp[w]
        numbplot = len(listdictdvrp)
        if numbplot == 0:
            figr, axis = plt.subplots(figsize=(8.25, 3.5), constrained_layout=True)
            axis.text(0.5, 0.5, 'No diagnostics available', ha='center', va='center')
            axis.axis('off')
            tdpy.save_figure(figr, pathplot, dpi=150, close_figure=True)
            continue
        numbcolr = min(2, numbplot)
        numbrows = math.ceil(numbplot / numbcolr)
        figr, axis = plt.subplots(
            numbrows,
            numbcolr,
            figsize=(8.25, 3.5 * numbrows),
            constrained_layout=True,
            squeeze=False,
        )
        listaxis = axis.ravel()
        for axisthis, dictdvrp in zip(listaxis, listdictdvrp):
            print('Reading from %s...' % dictdvrp['path'])
            axisthis.imshow(plt.imread(dictdvrp['path']))
            axisthis.axis('off')
        for axisthis in listaxis[numbplot:]:
            axisthis.axis('off')
        tdpy.save_figure(figr, pathplot, dpi=150, close_figure=True)

    return listpathdvrp


def setp_dvrp_output(gdat):
    """Populate DV report output paths on the workflow output dictionary."""

    listpathdvrp = make_dvrp_pages(gdat)
    gdat.dictmileoutp['listpathdvrp'] = listpathdvrp

    return listpathdvrp