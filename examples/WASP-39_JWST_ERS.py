import argparse

from miletos.diagnostics import run_simulated_jwst_transit_diagnostic
from miletos.paths import get_visuals_path

'''
This example uses an explicitly simulated WASP-39b-like benchmark to demonstrate
wavelength-resolved transit analysis for JWST observations.
'''

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run a simulated JWST transit diagnostic for WASP-39b.',
    )
    parser.add_argument('--typefileplot', choices=('png', 'pdf'), default='png')
    return parser.parse_args()

def main():
    arguments = parse_arguments()
    output_path = get_visuals_path() / (
        f'simulated_wasp39_jwst_diagnostic.{arguments.typefileplot}'
    )
    run_simulated_jwst_transit_diagnostic(output_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
