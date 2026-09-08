import os

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