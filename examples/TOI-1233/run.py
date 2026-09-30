#!/usr/bin/env python3
"""Analyze the observed TOI-1233 TESS photometry and PFS velocities, or reproduce Daylan et al. (2021a)."""

from tdpy.verbosity import print

import argparse

from miletos import daylan2021a
from miletos.toi1233 import run_toi1233_observation


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model',
        choices=('PlanetarySystem', 'PlanetarySystemWithTTVs'),
        default='PlanetarySystemWithTTVs',
    )
    parser.add_argument('--typefileplot', choices=('png', 'pdf'), default='png')
    parser.add_argument('--reproduce-daylan2021a', action='store_true',
                        help='Reproduce the four-planet TESS transit analysis of Daylan et al. (2021a) with PCAT.')
    parser.add_argument('--reuse', action='store_true', help='Replot a saved PCAT posterior instead of sampling.')
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    if arguments.reproduce_daylan2021a:
        samples, radi, state = daylan2021a.fit_daylan2021a_transits(
            numbchan=4, numbsamp=150000, numbburn=250000, boolreus=arguments.reuse)
        daylan2021a.plot_daylan2021a_transits(samples, radi, state, typefileplot=arguments.typefileplot)
        return 0
    result = run_toi1233_observation(
        typemodl=arguments.model,
        typefileplot=arguments.typefileplot,
    )
    print(
        f"Analyzed {result['strgtarg']} with {arguments.model} through the "
        'shared Miletos observational pipeline.'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())