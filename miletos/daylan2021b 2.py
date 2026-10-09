"""Miletos workflow for the WASP-121 b TESS phase-curve paper."""

from pathlib import Path

from . import main
from .paths import get_repository_path


def run_daylan2021b_reproduction(typefileplot: str = 'png', fit: bool = True) -> dict:
    """Analyze the public TESS Sector 7 light curve with Miletos."""
    if typefileplot not in {'png', 'pdf'}:
        raise ValueError("typefileplot must be 'png' or 'pdf'")

    example_path = get_repository_path() / 'examples' / 'WASP-121b'
    return main.init(
        strgmast='WASP-121',
        strgtarg='WASP-121b-Daylan2021b',
        labltarg='WASP-121 b',
        listtsecsele=[7],
        boolforcoffl=False,
        boolfitt=fit,
        boolplot=True,
        boolplottser=True,
        typefileplot=typefileplot,
        typeplotback='white',
        typeverb=0,
        liststrgtypedata=[['obsd'], []],
        listlablinst=[['TESS'], []],
        boolbdtr=[[False], []],
        typepriocomp='exar',
        dictfitt={'typemodl': 'PlanetarySystemEmittingCompanion'},
        pathtarg=str(example_path),
    )