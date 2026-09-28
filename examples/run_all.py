#!/usr/bin/env python3
"""Run every supported Miletos example and verify its plot."""

import argparse
import ast
import os
from pathlib import Path
import re
import subprocess
import sys


REPOSITORY_PATH = Path(__file__).parents[1]
QUICK_EXAMPLES = (
    ('simulated_transit_diagnostic.py', 'simulated_transit_diagnostic'),
    ('target_visibility.py', 'target_visibility_toi-1233'),
    ('WASP-39_JWST_ERS.py', 'simulated_wasp39_jwst_diagnostic'),
    ('examples.py', 'example_catalog_transit'),
)
CATALOG_ARGUMENTS = {
    'cnfg_WASP': (('18',), ('46',)),
    'cnfg_TOI_lists': (('MuSCAT2',), ('GUHJS2',)),
}
PLOT_PATH_PATTERN = re.compile(r'(?:Writing|Reading) to (.+\.(?:png|pdf))\.\.\.$')


def get_catalog_invocations():
    tree = ast.parse((REPOSITORY_PATH / 'examples' / 'examples.py').read_text())
    names = sorted(
        node.name for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith('cnfg_')
    )
    invocations = []
    for name in names:
        for arguments in CATALOG_ARGUMENTS.get(name, ((),)):
            invocations.append((name, arguments))
    return invocations


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--typefileplot', choices=('png', 'pdf'), default='png')
    parser.add_argument('--quick', action='store_true', help='Run only the four deterministic CI examples.')
    parser.add_argument('--list', action='store_true', help='List every catalog invocation without running it.')
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
    catalog_invocations = get_catalog_invocations()
    if arguments.list:
        for name, values in catalog_invocations:
            print(' '.join((name, *values)))
        return 0

    environment = os.environ.copy()
    environment.setdefault('MILETOS_PATH', str(REPOSITORY_PATH))
    output_root = Path(environment['MILETOS_PATH']) / 'visuals'

    for script_name, output_stem in QUICK_EXAMPLES:
        output_path = output_root / f'{output_stem}.{arguments.typefileplot}'
        if output_path.exists():
            print(f'Removing cached example output {output_path}...')
            output_path.unlink()
        command = [sys.executable, str(REPOSITORY_PATH / 'examples' / script_name)]
        if script_name == 'examples.py':
            command.extend(['cnfg_simulated_transit_diagnostic', arguments.typefileplot])
        else:
            command.extend(['--typefileplot', arguments.typefileplot])
        run_command(command, environment)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(f'Example did not produce a plot at {output_path}')
        print(f'Verified {output_path}...')

    if not arguments.quick:
        for name, values in catalog_invocations:
            command = [sys.executable, str(REPOSITORY_PATH / 'examples' / 'examples.py'), name, *values]
            print(f"Running {' '.join((name, *values))}...")
            output = run_command(command, environment)
            plot_paths = [Path(match.group(1)) for line in output.splitlines() if (match := PLOT_PATH_PATTERN.search(line))]
            if not any(path.is_file() and path.stat().st_size > 0 for path in plot_paths):
                raise RuntimeError(f'{name} did not report an existing plot')
            print(f'Verified {name} ({len(plot_paths)} plot references)...')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())