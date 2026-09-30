"""Observational TESS and PFS analysis of TOI-1233."""

from tdpy.verbosity import print

import os

import numpy as np

from . import main
from .paths import get_data_path, get_repository_path
from .pipeline import run_observational_pipeline


RIGHT_ASCENSION_DEGREES = 186.574548  # [deg]
DECLINATION_DEGREES = -51.362837  # [deg]


def _load_toi1233_time_series() -> dict[str, list]:
    """Load the published TOI-1233 TESS photometry and PFS velocities."""

    data_path = get_data_path() / 'TOI-1233' / 'data'
    velocity_path = data_path / 'HD108236_PFS_20220627.vels'
    print(f'Reading from {velocity_path}...')
    velocity = np.loadtxt(velocity_path)

    photometry_path = data_path / 'TESS_PDCSAP_FLUX_Sector10.csv'
    print(f'Reading from {photometry_path}...')
    photometry = np.loadtxt(photometry_path, delimiter=',')[:, None, :]

    radial_velocity = np.empty((velocity.shape[0], 1, 3))
    radial_velocity[:, 0, :] = velocity[:, :3]
    return {'Raw': [[[photometry]], [[radial_velocity]]]}


def analyze_toi1233(
    typemodl: str = 'PlanetarySystemWithTTVs',
    typefileplot: str = 'png',
) -> dict:
    """Run the standard Miletos time-series analysis for TOI-1233."""

    valid_models = {'PlanetarySystem', 'PlanetarySystemWithTTVs'}
    if typemodl not in valid_models:
        raise ValueError(f'Unknown TOI-1233 model: {typemodl}')
    target_path = get_repository_path() / 'examples' / 'TOI-1233'
    return main.init(
        pathtarg=os.path.join(str(target_path), ''),
        rasctarg=RIGHT_ASCENSION_DEGREES,
        decltarg=DECLINATION_DEGREES,
        labltarg='TOI-1233',
        strgtarg='TOI-1233',
        dictfitt={'typemodl': typemodl},
        strgexar='HD 108236',
        strgcnfg=typemodl,
        listarrytser=_load_toi1233_time_series(),
        liststrgtypedata=[['inpt'], ['inpt']],
        listlablinst=[['TESS'], ['PFS']],
        boolbdtr=[[False], [False]],
        boolforcoffl=True,
        boolfitt=False,
        typepriocomp='exar',
        boolplotpopl=False,
        typefileplot=typefileplot,
        typeplotback='white',
    )


def run_toi1233_observation(
    typemodl: str = 'PlanetarySystemWithTTVs',
    typefileplot: str = 'png',
) -> dict:
    """Analyze and plot TOI-1233 through the shared observational pipeline."""

    output_path = (
        get_repository_path()
        / 'examples'
        / 'TOI-1233'
        / typemodl
        / 'visuals'
    )
    return run_observational_pipeline(
        analyzer=analyze_toi1233,
        analysis_kwargs={
            'typemodl': typemodl,
            'typefileplot': typefileplot,
        },
        output_path=output_path,
        typefileplot=typefileplot,
    )