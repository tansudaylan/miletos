#!/usr/bin/env python3
"""Inspect observed TESS photometry of the WD 1856+534 system with Miletos."""

import argparse

import miletos
from tdpy.cli import add_plot_arguments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    miletos.init(
        strgmast='WD 1856+534',
        listlablinst=[['TESS'], []],
        liststrgtypedata=[['obsd'], []],
        dictfitt={'typemodl': 'PlanetarySystem'},
        typepriocomp='exar',
        boolfoldprio=False,
        boolfitt=False,
        typefileplot=arguments.typefileplot,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())