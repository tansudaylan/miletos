#!/usr/bin/env python3
"""Demonstrate PCAT sampling a synthetic NIRSpec HDO/H2O retrieval."""

from tdpy.verbosity import print

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from tdpy.cli import add_plot_arguments
from tdpy.plotting import save_figure
from miletos.paths import get_data_path, get_repository_path

from miletos.retrieval import sample_nirspec_retrieval


def synthetic_hdo_transmission_model(parameters, wavelength_microns):
    """Return an explicitly synthetic spectrum with water and HDO-shaped bands."""
    depth_offset_ppm, log10_hdo_h2o = parameters
    water = 320.0 * np.exp(-0.5 * ((wavelength_microns - 4.30) / 0.08) ** 2)
    hdo = 100.0 * 10.0 ** (log10_hdo_h2o + 5.0) * np.exp(
        -0.5 * ((wavelength_microns - 4.12) / 0.035) ** 2
    )
    return 20000.0 + depth_offset_ppm + water + hdo


def run_example(typefileplot="png", burn_samples=1000, retained_samples=2000, seed=188,
                *, time_tag=None, typeplotback="white"):
    """Recover an injected isotope ratio from synthetic NIRSpec-like bins."""
    if typefileplot not in ("png", "pdf"):
        raise ValueError("typefileplot must be png or pdf")
    if typeplotback not in ("white", "dark"):
        raise ValueError("typeplotback must be white or dark")
    time_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") if time_tag is None else time_tag
    if not isinstance(time_tag, str) or not re.fullmatch(r"[0-9]{8}_[0-9]{6}", time_tag):
        raise ValueError("time_tag must use YYYYMMDD_HHMMSS")
    datetime.strptime(time_tag, "%Y%m%d_%H%M%S")
    output_path = (get_repository_path() / "examples" / "nirspec_isotope_retrieval"
                   / "visuals" / f"nirspec_isotope_injection_recovery_{time_tag}.{typefileplot}")
    posterior_path = get_data_path() / "nirspec_isotope_retrieval" / f"posterior_{time_tag}.npz"
    metadata_path = posterior_path.with_suffix(".json")
    for path in (output_path, posterior_path, metadata_path):
        if path.exists():
            raise FileExistsError(path.name)
    source_path = Path(__file__).resolve()
    print(f"Reading from {source_path}...")
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    wavelength_microns = np.linspace(3.85, 4.60, 96)  # [micron]
    injected_parameters = np.array([12.0, -5.0])  # [ppm, log10(HDO/H2O)]
    uncertainty_ppm = 12.0  # [ppm]
    correlation = np.exp(
        -np.abs(wavelength_microns[:, None] - wavelength_microns[None, :]) / 0.04
    )
    covariance_ppm2 = uncertainty_ppm**2 * correlation  # [ppm^2]
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
    print(f"Reading from {source_path}...")
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("Example source changed during sampling; refusing to publish outputs")

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
        color="black",
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
    axes[0].set_xlabel("Wavelength [micrometers]")
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

    save_figure(figure, output_path, typefileplot=typefileplot, typeplotback=typeplotback,
                close_figure=True)
    posterior_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {posterior_path}...")
    np.savez_compressed(
        posterior_path, samples=samples, log_probability=log_probability,
        wavelength_microns=wavelength_microns, observed_depth_ppm=observed_depth_ppm,
        covariance_ppm2=covariance_ppm2, injected_parameters=injected_parameters,
        lower_model=lower_model, median_model=median_model, upper_model=upper_model,
    )
    metadata = {
        "data_kind": "Explicitly synthetic Gaussian features, not physical isotope opacities or observations",
        "time_tag_utc": time_tag, "seed": seed, "chain_count": 1,
        "burn_samples": burn_samples, "retained_samples": int(samples.shape[0]),
        "parameter_names": ["depth_offset_ppm", "log10_hdo_h2o"],
        "posterior_quantiles_16_50_84": np.percentile(samples, [16, 50, 84], axis=0).tolist(),
        "source_sha256": source_hash, "figure": output_path.name,
        "posterior": posterior_path.name,
        "interpretation": "Single-chain synthetic integration example; convergence and real isotope sensitivity are unverified",
    }
    print(f"Writing to {metadata_path}...")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
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
    parser.add_argument("--time-tag")
    parser.add_argument("--typeplotback", choices=("white", "dark"), default="white")
    arguments = parser.parse_args()
    run_example(arguments.typefileplot, arguments.burn_samples, arguments.retained_samples, arguments.seed,
                time_tag=arguments.time_tag, typeplotback=arguments.typeplotback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())