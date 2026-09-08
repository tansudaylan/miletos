import os

import matplotlib.pyplot as plt


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

        figr = plt.figure(figsize=(8.25, 11.75))
        for dictdvrp in gdat.listdictdvrp[w]:
            axis = figr.add_axes(dictdvrp['limt'])
            print('Reading from %s...' % dictdvrp['path'])
            axis.imshow(plt.imread(dictdvrp['path']))
            axis.axis('off')
        if gdat.typeverb > 0:
            print('Writing to %s...' % pathplot)
        plt.savefig(pathplot, dpi=600)
        plt.close()

    return listpathdvrp


def setp_dvrp_output(gdat):
    """Populate DV report output paths on the workflow output dictionary."""

    listpathdvrp = make_dvrp_pages(gdat)
    gdat.dictmileoutp['listpathdvrp'] = listpathdvrp

    return listpathdvrp