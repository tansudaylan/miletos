#!/usr/bin/env python3
"""Analyze the observed TOI-1233 TESS photometry and PFS velocities."""

import argparse

from miletos.daylan2021 import run_daylan2021_reproduction
from miletos.toi1233 import run_toi1233_observation


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model',
        choices=('PlanetarySystem', 'PlanetarySystemWithTTVs'),
        default='PlanetarySystemWithTTVs',
    )
    parser.add_argument(
        '--reproduce-daylan2021',
        action='store_true',
        help='recover the four-planet Daylan et al. (2021) TESS results',
    )
    parser.add_argument('--typefileplot', choices=('png', 'pdf'), default='png')
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    if arguments.reproduce_daylan2021:
        result = run_daylan2021_reproduction(typefileplot=arguments.typefileplot)
        print(
            'Daylan et al. (2021) four-planet recovery: '
            f'maximum difference = {result.maximum_sigma_difference:.2f} sigma; '
            f'passed = {result.reproduces_published_results}.'
        )
        return 0 if result.reproduces_published_results else 1
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