import fnmatch
import os
from pathlib import Path

import numpy as np

import tdpy


PATH_ENV_VAR = "MILETOS_PATH"


def get_repository_path() -> Path:
    """Return the repository path configured by MILETOS_PATH."""

    path_value = os.environ.get(PATH_ENV_VAR)
    if not path_value or not path_value.strip():
        raise EnvironmentError(f"{PATH_ENV_VAR} is required and cannot be empty.")
    return Path(path_value).expanduser().resolve()


def get_data_path() -> Path:
    """Return the ignored repository-local data directory."""

    return get_repository_path() / "data"


def get_visuals_path() -> Path:
    """Return the ignored repository-local visualization directory."""

    return get_repository_path() / "visuals"


def retr_tsecpathlocl(tici, typeverb=1):
    """Return local SPOC-sector availability for a TESS target."""

    pathbase = os.path.join(tdpy.retr_pathbase('tess'), 'data', 'lcur')
    path = os.path.join(pathbase, 'tsec', 'tsec_spoc_%016d.csv' % tici)
    if not os.path.exists(path):
        listtsecsele = np.arange(1, 60)
        listpath = []
        listtsec = []
        strgtagg = '*-%016d-*.fits' % tici
        for tsec in listtsecsele:
            pathtemp = os.path.join(pathbase, 'sector-%02d' % tsec) + '/'
            listpathtemp = fnmatch.filter(os.listdir(pathtemp), strgtagg)

            if len(listpathtemp) > 0:
                listpath.append(pathtemp + listpathtemp[0])
                listtsec.append(tsec)

        listtsec = np.array(listtsec).astype(int)
        print('Writing to %s...' % path)
        with open(path, 'w') as objtfile:
            for k in range(len(listpath)):
                objtfile.write('%d,%s\n' % (listtsec[k], listpath[k]))
    else:
        if typeverb > 0:
            print('Reading from %s...' % path)
        with open(path, 'r') as objtfile:
            listtsec = []
            listpath = []
            for line in objtfile:
                linesplt = line.split(',')
                listtsec.append(linesplt[0])
                listpath.append(linesplt[1][:-1])
        listtsec = np.array(listtsec).astype(int)

    return listtsec, listpath


def setp_base_paths(gdat):
    """Populate shared base paths on the runtime state."""

    gdat.pathbasemile = tdpy.retr_pathbase('miletos')
    if gdat.pathbase is None:
        gdat.pathbase = gdat.pathbasemile
    gdat.pathbaselygo = tdpy.retr_pathbase('lygos')


def chec_path_input(gdat):
    """Return whether the supplied path inputs are mutually consistent."""

    boolvalid = (
        gdat.pathtarg is None
        and gdat.pathbase is None
        and gdat.pathdatatarg is None
        and gdat.pathvisutarg is None
    ) or (
        gdat.pathtarg is not None
        and gdat.pathbase is None
        and gdat.pathdatatarg is None
        and gdat.pathvisutarg is None
    ) or (
        gdat.pathtarg is None
        and gdat.pathbase is not None
        and gdat.pathdatatarg is None
        and gdat.pathvisutarg is None
    ) or (
        gdat.pathtarg is None
        and gdat.pathbase is None
        and gdat.pathdatatarg is not None
        and gdat.pathvisutarg is not None
    )

    return boolvalid


def setp_target_paths(gdat):
    """Populate target-specific data and visualization paths on the runtime state."""

    if gdat.strgcnfg is None or gdat.strgcnfg == '':
        strgcnfgtemp = ''
    else:
        strgcnfgtemp = gdat.strgcnfg + '/'

    gdat.pathtargcnfg = gdat.pathtarg + strgcnfgtemp

    if gdat.booldiag and gdat.pathtargcnfg.endswith('//'):
        print('')
        print('')
        print('')
        print('gdat.pathtargcnfg')
        print(gdat.pathtargcnfg)
        print('strgcnfgtemp')
        print(strgcnfgtemp)
        raise Exception('')

    gdat.pathdatatarg = gdat.pathtargcnfg + 'data/'
    gdat.pathvisutarg = gdat.pathtargcnfg + 'visuals/'


def setp_mast_path(gdat):
    """Populate the MAST download/cache directory on the runtime state."""

    gdat.pathdatamast = tdpy.retr_pathbase('mast')


def setp_feature_paths(gdat):
    """Populate feature-visualization directories on the runtime state."""

    gdat.pathvisufeat = gdat.pathvisutarg + 'feat/'

    for strgpdfn in gdat.liststrgpdfn:
        pathvisupdfn = gdat.pathvisufeat + strgpdfn + '/'
        setattr(gdat, 'pathvisufeatplan' + strgpdfn, pathvisupdfn + 'featplan/')
        setattr(gdat, 'pathvisufeatsyst' + strgpdfn, pathvisupdfn + 'featsyst/')
        setattr(gdat, 'pathvisudataplan' + strgpdfn, pathvisupdfn + 'dataplan/')


def setp_alle_path(gdat, typemodl):
    """Populate and ensure the allesfitter run directory for a model."""

    gdat.pathalle[typemodl] = tdpy.ensr_path(gdat.pathallebase + 'allesfit_%s/' % typemodl)


def ensr_gdat_paths(gdat):
    """Ensure directory-like path attributes on the runtime state exist."""

    for attr, valu in gdat.__dict__.items():
        if attr.startswith('path') and valu is not None and not isinstance(valu, dict) and valu.endswith('/'):
            setattr(gdat, attr, tdpy.ensr_path(valu))