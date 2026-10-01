"""Search the archived Sector 10 TOI validation subset using cached QLP curves."""

import argparse
from tdpy.verbosity import print

from run_pilot import run_sector_validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true",
                        help="retrieve any missing QLP FITS products from MAST")
    arguments = parser.parse_args()
    targets, results, products = run_sector_validation(download=arguments.download)
    print(f"Selected unique TIC targets: {len(targets)}")
    print(f"QLP products searched: {int(results['searched'].sum())}")
    print(f"Initial-pass candidates: {int(results['detection'].sum())}")
    print(f"Review cases: {int(results['disposition'].eq('review').sum())}")
    print(f"Products: {products}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
