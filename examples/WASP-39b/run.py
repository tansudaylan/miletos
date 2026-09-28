import argparse

from miletos.ers import run_wasp39_ers_g395h_reproduction

'''
This example reproduces the observational NIRSpec G395H analysis accompanying
Alderson et al. (2023) from its public Zenodo data products.
'''

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Reproduce the published WASP-39b JWST ERS G395H analysis.',
    )
    parser.add_argument('--typefileplot', choices=('png', 'pdf'), default='png')
    parser.add_argument(
        '--refresh-data',
        action='store_true',
        help='Download a fresh copy of the public Zenodo data archive.',
    )
    return parser.parse_args()

def main():
    arguments = parse_arguments()
    result = run_wasp39_ers_g395h_reproduction(
        typefileplot=arguments.typefileplot,
        refresh_data=arguments.refresh_data,
    )
    print(
        f'Reproduced {result.wavelength_microns.size} published wavelength bins; '
        f'mean uncertainty = {result.transit_depth_uncertainty_ppm.mean():.0f} ppm; '
        f'reduced chi-squared = {result.reduced_chi_squared:.2f}.'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
