#!/usr/bin/env python3
"""Reproduce the TESS phase-curve analysis of WASP-121 b by Daylan et al. (2021b)."""

import argparse
from tdpy.cli import add_plot_arguments

from miletos import daylan2021b


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_arguments(parser)
    parser.add_argument(
        '--prepare-only',
        action='store_true',
        help='Retrieve, reduce, and phase-fold the observations without fitting.',
    )
    parser.add_argument('--reuse', action='store_true', help='Replot a saved PCAT posterior instead of sampling.')
    arguments = parser.parse_args()
    daylan2021b.run_daylan2021b_reproduction(typefileplot=arguments.typefileplot, fit=False)
    if not arguments.prepare_only:
        samples, tmpt, state = daylan2021b.fit_daylan2021b_phase_curve(
            numbchan=4, numbsamp=100000, numbburn=100000, boolreus=arguments.reuse)
        daylan2021b.plot_daylan2021b_phase_curve(samples, tmpt, state, typefileplot=arguments.typefileplot)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
