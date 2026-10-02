#!/usr/bin/env python3
"""Inspect observed TESS photometry of TRAPPIST-1 with Miletos."""

import argparse

import miletos
from tdpy.cli import add_plot_arguments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    miletos.init(
        strgmast='TRAPPIST-1',
        listlablinst=[['TESS'], []],
        liststrgtypedata=[['obsd'], []],
        dictfitt={'typemodl': 'PlanetarySystemWithTTVs'},
        typelcurtpxftess='SPOC',
        boolfitt=False,
        typefileplot=arguments.typefileplot,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())