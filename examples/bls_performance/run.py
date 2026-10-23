#!/usr/bin/env python3
"""Measure Miletos BLS runtime on an explicitly synthetic transit."""

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tdpy.cli import add_plot_arguments
from tdpy.plotting import plot_style, save_plot
from tdpy.verbosity import print

from miletos.box_least_squares import search_box_least_squares


INJECTED_PERIOD_DAYS = 3.2  # [day]
INJECTED_EPOCH_DAYS = 0.4  # [day]
INJECTED_DURATION_DAYS = 0.12  # [day]
INJECTED_DEPTH = 0.01
BASELINE_DAYS = 27.0  # [day]
VISUAL_DIRECTORY = Path(__file__).resolve().parent / "visuals"


def make_synthetic_light_curve(cadence_count: int, seed: int = 824):
    """Return a seeded 27-day light curve with a 1% box-shaped transit."""

    if cadence_count < 3:
        raise ValueError("cadence_count must be at least three")
    time_days = np.linspace(0.0, BASELINE_DAYS, cadence_count)
    phase_days = np.remainder(
        time_days - INJECTED_EPOCH_DAYS + 0.5 * INJECTED_PERIOD_DAYS,
        INJECTED_PERIOD_DAYS,
    ) - 0.5 * INJECTED_PERIOD_DAYS
    flux = np.ones(cadence_count)
    flux[np.abs(phase_days) < 0.5 * INJECTED_DURATION_DAYS] -= INJECTED_DEPTH
    flux += np.random.default_rng(seed).normal(0.0, 0.001, cadence_count)
    flux_uncertainty = np.full(cadence_count, 0.001)
    return time_days, flux, flux_uncertainty


def run_benchmark(
    cadence_counts=(1000, 3000, 6000),
    frequency_factors=(1.0, 2.0, 4.0),
    *,
    output_directory: str | Path = VISUAL_DIRECTORY,
    typefileplot: str = "png",
    typeplotback: str = "white",
) -> pd.DataFrame:
    """Time the real Miletos BLS search over cadence and frequency-grid sizes."""

    cadence_counts = tuple(int(value) for value in cadence_counts)
    frequency_factors = tuple(float(value) for value in frequency_factors)
    if not cadence_counts or any(value < 3 for value in cadence_counts):
        raise ValueError("cadence_counts must contain values of at least three")
    if not frequency_factors or not np.isfinite(frequency_factors).all() or any(
        value <= 0.0 for value in frequency_factors
    ):
        raise ValueError("frequency_factors must contain finite positive values")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for cadence_count in cadence_counts:
        time_days, flux, flux_uncertainty = make_synthetic_light_curve(cadence_count)
        for frequency_factor in frequency_factors:
            start_time = perf_counter()
            result = search_box_least_squares(
                time_days,
                flux,
                flux_uncertainty,
                duration_days=(0.08, 0.12, 0.16),  # [day]
                minimum_period_days=0.5,  # [day]
                maximum_period_days=10.0,  # [day]
                frequency_factor=frequency_factor,
            )
            elapsed_seconds = perf_counter() - start_time  # [s]
            rows.append({
                "cadence_count": cadence_count,
                "frequency_factor": frequency_factor,
                "elapsed_seconds": elapsed_seconds,
                "recovered_period_days": result["period_days"],  # [day]
                "depth_snr": result["depth_snr"],
                "injected_period_days": INJECTED_PERIOD_DAYS,  # [day]
            })

    results = pd.DataFrame(rows)
    table_path = output_directory / "bls_runtime_benchmark.csv"
    print(f"Writing to {table_path}...")
    results.to_csv(table_path, index=False)

    with plot_style(typeplotback):
        figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
        for frequency_factor, group in results.groupby("frequency_factor", sort=True):
            group = group.sort_values("cadence_count")
            axes[0].plot(
                group["cadence_count"],
                group["elapsed_seconds"],
                marker="o",
                label=f"frequency factor {frequency_factor:g}",
            )
        for cadence_count, group in results.groupby("cadence_count", sort=True):
            group = group.sort_values("frequency_factor")
            axes[1].plot(
                group["frequency_factor"],
                group["elapsed_seconds"],
                marker="o",
                label=f"{cadence_count:,} cadences",
            )
        axes[0].set(xlabel="Cadence count", ylabel="Search time [s]", title="Runtime versus data size")
        axes[1].set(xlabel="BLS frequency factor", ylabel="Search time [s]", title="Runtime versus frequency factor")
        for axis in axes:
            axis.grid(False)
            axis.legend(frameon=False)

        figure_path = output_directory / f"bls_runtime_benchmark.{typefileplot}"
        print(f"Writing to {figure_path}...")
        save_plot(figure, figure_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_arguments(parser)
    parser.add_argument("--cadences", type=int, nargs="+", default=(1000, 3000, 6000))
    parser.add_argument("--frequency-factors", type=float, nargs="+", default=(1.0, 2.0, 4.0))
    parser.add_argument("--output-directory", type=Path, default=VISUAL_DIRECTORY)
    parser.add_argument(
        "--typeplotback",
        choices=("white", "dark", "norm", "black"),
        default="white",
    )
    arguments = parser.parse_args(argv)
    results = run_benchmark(
        arguments.cadences,
        arguments.frequency_factors,
        output_directory=arguments.output_directory,
        typefileplot=arguments.typefileplot,
        typeplotback=arguments.typeplotback,
    )
    print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())