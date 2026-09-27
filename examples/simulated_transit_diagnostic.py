#!/usr/bin/env python3
"""Run the deterministic Miletos simulated-transit diagnostic."""

import os
from pathlib import Path

from miletos.diagnostics import run_simulated_transit_diagnostic
from tdpy.cli import parse_plot_arguments


def parse_arguments():
    return parse_plot_arguments(
        description="Run a deterministic Miletos simulated-transit diagnostic."
    )


def main() -> int:
    arguments = parse_arguments()
    repository_path = Path(os.environ["MILETOS_PATH"])
    output_path = repository_path / "visuals" / (
        f"simulated_transit_diagnostic.{arguments.typefileplot}"
    )
    run_simulated_transit_diagnostic(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())