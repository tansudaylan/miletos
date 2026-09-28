#!/usr/bin/env python3
"""Analyze the observed TOI-1233 TESS photometry and PFS velocities."""

import argparse

from miletos.toi1233 import run_toi1233_observation


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model',
        choices=('PlanetarySystem', 'PlanetarySystemWithTTVs'),
        default='PlanetarySystemWithTTVs',
    )
    parser.add_argument('--typefileplot', choices=('png', 'pdf'), default='png')
    return parser.parse_args()


def main():
    arguments = parse_arguments()
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