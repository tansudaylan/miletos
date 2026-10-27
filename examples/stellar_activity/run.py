#!/usr/bin/env python3
"""Fit a seeded synthetic flare light curve with Miletos and Ephesos."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ephesos import evaluate_spot_flare_model
from miletos import fit_spot_flare_light_curve, parse_spot_flare_parameters
from miletos.paths import get_example_visuals_path
from tdpy.cli import parse_plot_arguments
from tdpy.verbosity import print


def run_example(output_path):
    time_days = np.linspace(0.0, 3.0, 301)  # [day]
    true_parameters = np.array([0.3, 0.3, 10.0, 90.0, 0.0, 0.001, 1.5, 0.03, 0.06, 0.3])
    model_parameters = parse_spot_flare_parameters(true_parameters, number_spots=0, number_flares=1)
    true_flux = evaluate_spot_flare_model(time_days, **model_parameters)
    uncertainty = np.full(time_days.size, 0.0003)
    flux = true_flux + np.random.default_rng(42).normal(size=time_days.size) * uncertainty
    initial = [0.3, 0.3, 10.0, 90.0, 0.0, 0.0, 1.48, 0.025, 0.08, 0.25]
    bounds = (
        [0.0, 0.0, 1.0, 0.0, -0.5, -0.01, 1.0, 0.001, 0.01, 0.05],
        [1.0, 1.0, 20.0, 180.0, 0.5, 0.01, 2.0, 0.1, 0.2, 0.8],
    )
    result = fit_spot_flare_light_curve(
        time_days, flux, uncertainty, initial, number_spots=0, number_flares=1, bounds=bounds,
    )
    figure, axes = plt.subplots(2, 1, figsize=(8.0, 5.6), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]}, facecolor="white")
    axes[0].errorbar(time_days, flux, yerr=uncertainty, fmt=".", color="0.6", markersize=3,
                     linewidth=0.5, label="Seeded synthetic data")
    axes[0].plot(time_days, true_flux, color="#A51417", linestyle="--", label="Input model")
    axes[0].plot(time_days, result["model_flux"], color="#007A6F", label="Miletos local fit")
    axes[0].set(ylabel="Relative flux", title="Synthetic stellar flare fit")
    axes[0].legend(frameon=True, fancybox=True, framealpha=1.0)
    axes[1].plot(time_days, flux - result["model_flux"], ".", color="black", markersize=3)
    axes[1].axhline(0.0, color="0.5", linewidth=0.8)
    axes[1].set(xlabel="Elapsed time [day]", ylabel="Residual", xlim=(0.0, 3.0))
    for axis in axes:
        axis.grid(False)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {"time_days": time_days, "flux": flux, "uncertainty": uncertainty,
            "true_parameters": true_parameters, **result}


def main():
    arguments = parse_plot_arguments(description=__doc__)
    output_path = get_example_visuals_path("stellar_activity")
    run_example(output_path / f"stellar_activity.{arguments.typefileplot}")
    return 0


if __name__ == "__main__":
    main()