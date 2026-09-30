#!/usr/bin/env python3
"""Run the Miletos TESS phase-curve analysis of WASP-121 b."""

import argparse

from miletos.daylan2021b import run_daylan2021b_reproduction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--typefileplot', choices=('png', 'pdf'), default='png')
    parser.add_argument(
        '--prepare-only',
        action='store_true',
        help='Retrieve, reduce, and phase-fold the observations without fitting.',
    )
    arguments = parser.parse_args()
    run_daylan2021b_reproduction(
        typefileplot=arguments.typefileplot,
        fit=not arguments.prepare_only,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())