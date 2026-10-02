#!/usr/bin/env python3
"""Run every supported Miletos example and verify its plot."""

from tdpy.verbosity import print

import argparse
from tdpy.cli import add_plot_arguments
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_PATH = Path(__file__).parents[1]
QUICK_EXAMPLES = (
    (
        'simulated_transit/run.py',
        'examples/simulated_transit/visuals/simulated_transit_diagnostic',
    ),
    (
        'WASP-39b/run.py',
        'examples/WASP-39b/visuals/wasp39_ers_g395h_transmission_spectrum',
    ),
)
OBSERVATIONAL_EXAMPLES = (
    'TOI-1233/run.py',
    'WASP-121b/run.py',
    'WD1856b/run.py',
    'TRAPPIST-1/run.py',
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_arguments(parser)
    parser.add_argument('--quick', action='store_true', help='Run only the two CI examples.')
    parser.add_argument('--list', action='store_true', help='List the observational target examples without running them.')
    return parser.parse_args()


def run_command(command, environment):
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_PATH,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    print(completed.stdout, end='')
    print(completed.stderr, end='', file=sys.stderr)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed.stdout


def main():
    arguments = parse_arguments()
    if arguments.list:
        for script_name in OBSERVATIONAL_EXAMPLES:
            print(script_name)
        return 0

    environment = os.environ.copy()
    environment.setdefault('MILETOS_PATH', str(REPOSITORY_PATH))
    output_root = Path(environment['MILETOS_PATH'])

    for script_name, output_stem in QUICK_EXAMPLES:
        output_path = output_root / f'{output_stem}.{arguments.typefileplot}'
        if output_path.exists():
            print(f'Removing cached example output {output_path}...')
            output_path.unlink()
        command = [sys.executable, str(REPOSITORY_PATH / 'examples' / script_name)]
        command.extend(['--typefileplot', arguments.typefileplot])
        run_command(command, environment)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(f'Example did not produce a plot at {output_path}')
        print(f'Verified {output_path}...')

    if not arguments.quick:
        for script_name in OBSERVATIONAL_EXAMPLES:
            command = [sys.executable, str(REPOSITORY_PATH / 'examples' / script_name)]
            command.extend(['--typefileplot', arguments.typefileplot])
            run_command(command, environment)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())