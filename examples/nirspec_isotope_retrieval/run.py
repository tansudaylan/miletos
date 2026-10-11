#!/usr/bin/env python3
"""Demonstrate PCAT sampling a synthetic NIRSpec HDO/H2O retrieval."""

from tdpy.verbosity import print

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tdpy.cli import add_plot_arguments
from miletos.paths import get_repository_path

from miletos.retrieval import sample_nirspec_retrieval


def synthetic_hdo_transmission_model(parameters, wavelength_microns):
    """Return an explicitly synthetic spectrum with water and HDO-shaped bands."""
    depth_offset_ppm, log10_hdo_h2o = parameters
    water = 320.0 * np.exp(-0.5 * ((wavelength_microns - 4.30) / 0.08) ** 2)
    hdo = 100.0 * 10.0 ** (log10_hdo_h2o + 5.0) * np.exp(
        -0.5 * ((wavelength_microns - 4.12) / 0.035) ** 2
    )
    return 20000.0 + depth_offset_ppm + water + hdo


def run_example(typefileplot="png", burn_samples=1000, retained_samples=2000, seed=188):
    """Recover an injected isotope ratio from synthetic NIRSpec-like bins."""
    wavelength_microns = np.linspace(3.85, 4.60, 96)
    injected_parameters = np.array([12.0, -5.0])  # [ppm, log10(HDO/H2O)]
    uncertainty_ppm = 12.0
    correlation = np.exp(
        -np.abs(wavelength_microns[:, None] - wavelength_microns[None, :]) / 0.04
    )
    covariance_ppm2 = uncertainty_ppm**2 * correlation
    random = np.random.default_rng(seed)
    observed_depth_ppm = synthetic_hdo_transmission_model(
        injected_parameters, wavelength_microns
    ) + np.linalg.cholesky(covariance_ppm2) @ random.normal(size=wavelength_microns.size)

    chains, log_probability = sample_nirspec_retrieval(
        wavelength_microns,
        observed_depth_ppm,
        covariance_ppm2,
        synthetic_hdo_transmission_model,
        parameter_names=("depth_offset_ppm", "log10_hdo_h2o"),
        prior_types=("self", "self"),
        prior_minima=(-80.0, -7.0),
        prior_maxima=(80.0, -3.0),
        initial_values=injected_parameters,
        proposal_scales=(0.15, 0.08),
        chain_count=1,
        retained_samples=retained_samples,
        burn_samples=burn_samples,
        seed=seed,
        typeverb=0,
        booladaptstdp=False,
    )
    samples = chains.reshape(-1, chains.shape[-1])
    median_parameters = np.median(samples, axis=0)
    selected_samples = samples[
        np.linspace(0, samples.shape[0] - 1, min(200, samples.shape[0]), dtype=int)
    ]
    model_draws = np.asarray([
        synthetic_hdo_transmission_model(sample, wavelength_microns)
        for sample in selected_samples
    ])
    lower_model, median_model, upper_model = np.percentile(model_draws, [16.0, 50.0, 84.0], axis=0)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8.0, 5.4),
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    axes[0].errorbar(
        wavelength_microns,
        observed_depth_ppm,
        yerr=uncertainty_ppm,
        fmt="o",
        markersize=3,
        color="#4B5A60",
        label="Synthetic NIRSpec bins",
    )
    axes[0].fill_between(
        wavelength_microns,
        lower_model,
        upper_model,
        color="#A51417",
        alpha=0.22,
        label="PCAT 68% interval",
    )
    axes[0].plot(wavelength_microns, median_model, color="#A51417", label="PCAT median model")
    axes[0].set_ylabel("Transit depth [ppm]")
    axes[0].legend(frameon=False)
    axes[1].hist(samples[:, 1], bins=32, density=True, color="#007360", alpha=0.75)
    axes[1].axvline(
        injected_parameters[1], color="#A51417", linestyle="--", label="Injected log10(HDO/H2O)"
    )
    axes[1].axvline(median_parameters[1], color="black", label="PCAT posterior median")
    axes[1].set_xlabel("log10(HDO/H2O)")
    axes[1].set_ylabel("Posterior density")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(False)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Synthetic NIRSpec isotope injection recovery")
    figure.tight_layout()

    output_path = (
        get_repository_path()
        / "examples"
        / "nirspec_isotope_retrieval"
        / "visuals"
        / f"nirspec_isotope_injection_recovery.{typefileplot}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(output_path, dpi=300 if typefileplot == "png" else None, bbox_inches="tight")
    plt.close(figure)
    print(
        f"Synthetic log10(HDO/H2O): injected {injected_parameters[1]:.2f}, "
        f"posterior median {median_parameters[1]:.2f}; "
        f"finite posterior draws {np.isfinite(log_probability).sum()}"
    )
    return output_path, samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_arguments(parser)
    parser.add_argument("--burn-samples", type=int, default=1000)
    parser.add_argument("--retained-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=188)
    arguments = parser.parse_args()
    run_example(arguments.typefileplot, arguments.burn_samples, arguments.retained_samples, arguments.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())