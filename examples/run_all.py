#!/usr/bin/env python3
"""Run every supported Miletos example and verify its plot."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_PATH = Path(__file__).parents[1]
EXAMPLES = (
    ('simulated_transit_diagnostic.py', 'simulated_transit_diagnostic'),
    ('target_visibility.py', 'target_visibility_toi-1233'),
    ('WASP-39_JWST_ERS.py', 'simulated_wasp39_jwst_diagnostic'),
    ('examples.py', 'example_catalog_transit'),
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--typefileplot', choices=('png', 'pdf'), default='png')
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    environment = os.environ.copy()
    environment.setdefault('MILETOS_PATH', str(REPOSITORY_PATH))
    output_root = Path(environment['MILETOS_PATH']) / 'visuals'

    for script_name, output_stem in EXAMPLES:
        output_path = output_root / f'{output_stem}.{arguments.typefileplot}'
        if output_path.exists():
            print(f'Removing cached example output {output_path}...')
            output_path.unlink()
        command = [sys.executable, str(REPOSITORY_PATH / 'examples' / script_name)]
        if script_name == 'examples.py':
            command.extend(['cnfg_simulated_transit_diagnostic', arguments.typefileplot])
        else:
            command.extend(['--typefileplot', arguments.typefileplot])
        subprocess.run(command, cwd=REPOSITORY_PATH, env=environment, check=True)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(f'Example did not produce a plot at {output_path}')
        print(f'Verified {output_path}...')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())