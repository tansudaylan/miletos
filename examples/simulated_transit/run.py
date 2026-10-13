#!/usr/bin/env python3
"""Run the deterministic Miletos simulated-transit diagnostic."""

import os
from pathlib import Path
import argparse

from miletos.diagnostics import run_simulated_transit_diagnostic
from tdpy.cli import add_plot_arguments


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a deterministic Miletos simulated-transit diagnostic."
    )
    add_plot_arguments(parser)
    parser.add_argument("--bokeh", action="store_true", help="write an interactive HTML figure")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    repository_path = Path(os.environ["MILETOS_PATH"])
    file_type = "html" if arguments.bokeh else arguments.typefileplot
    output_path = repository_path / "examples" / "simulated_transit" / "visuals" / (
        f"simulated_transit_diagnostic.{file_type}"
    )
    run_simulated_transit_diagnostic(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())