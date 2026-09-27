"""Compact, reproducible diagnostics built from the Miletos pipeline APIs."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np

import ephesos

from .main import bdtr_tser, fold_tser, rebn_tser


@dataclass(frozen=True)
class TransitDiagnostic:
    """Input, intermediate, and output arrays for a simulated transit analysis."""

    time_days: np.ndarray
    observed_flux: np.ndarray
    baseline_flux: np.ndarray
    detrended_flux: np.ndarray
    model_flux: np.ndarray
    phase: np.ndarray
    folded_flux: np.ndarray
    folded_model_flux: np.ndarray
    binned_phase: np.ndarray
    binned_flux: np.ndarray
    binned_uncertainty: np.ndarray
    period_days: float


def analyze_simulated_transit(seed: int = 7) -> TransitDiagnostic:
    """Run detrending, phase folding, and binning on a simulated transit series."""

    time_days = np.arange(0.0, 27.0, 10.0 / (24.0 * 60.0))
    period_days = 3.2
    epoch_days = 0.4
    uncertainty = np.full(time_days.size, 250e-6)

    model_flux = ephesos.eval_modl(
        time_days,
        "PlanetarySystem",
        pericomp=np.array([period_days]),
        epocmtracomp=np.array([epoch_days]),
        rsmacomp=np.array([0.085]),
        cosicomp=np.array([0.0]),
        rratcomp=np.array([0.085]),
        typelmdk="quad",
        booldiag=False,
        typeverb=0,
    )["rflx"][:, 0]
    baseline_flux = (
        1.0
        + 1_500e-6 * np.sin(2.0 * np.pi * time_days / 8.0)
        + 600e-6 * (time_days - np.mean(time_days)) / np.ptp(time_days)
    )
    observed_flux = baseline_flux * model_flux
    observed_flux += np.random.default_rng(seed).normal(0.0, uncertainty)

    detrended_flux = bdtr_tser(
        time_days,
        observed_flux,
        uncertainty,
        epocmask=np.array([epoch_days]),
        perimask=np.array([period_days]),
        duramask=np.array([5.0]),
        boolbrekregi=False,
        booladdddiscbdtr=False,
        typebdtr="Spline",
        timescalbdtr=1.5,
        typeverb=0,
    )[0]
    estimated_baseline = observed_flux - detrended_flux + 1.0

    detrended_series = np.column_stack((time_days, detrended_flux, uncertainty))
    model_series = np.column_stack((time_days, model_flux, np.zeros(time_days.size)))
    folded_series = fold_tser(detrended_series, epoch_days, period_days)
    folded_model = fold_tser(model_series, epoch_days, period_days)
    binned_series = rebn_tser(folded_series, numbbins=65)
    finite_bins = np.isfinite(binned_series[:, 1])

    return TransitDiagnostic(
        time_days=time_days,
        observed_flux=observed_flux,
        baseline_flux=estimated_baseline,
        detrended_flux=detrended_flux,
        model_flux=model_flux,
        phase=folded_series[:, 0],
        folded_flux=folded_series[:, 1],
        folded_model_flux=folded_model[:, 1],
        binned_phase=binned_series[finite_bins, 0],
        binned_flux=binned_series[finite_bins, 1],
        binned_uncertainty=binned_series[finite_bins, 2],
        period_days=period_days,
    )


def plot_transit_diagnostic(result: TransitDiagnostic, output_path: Path) -> Path:
    """Plot the input, detrending stage, and phase-folded pipeline output."""

    if output_path.suffix not in {".png", ".pdf"}:
        raise ValueError("output_path must end in .png or .pdf")

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(8.0, 8.2),
        constrained_layout=True,
        facecolor="white",
    )
    for axis in axes:
        axis.set_facecolor("white")
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].plot(
        result.time_days,
        1e6 * (result.observed_flux - 1.0),
        ".",
        color="#64818A",
        markersize=1.6,
        alpha=0.65,
        label="Simulated measurements",
    )
    axes[0].plot(
        result.time_days,
        1e6 * (result.baseline_flux - 1.0),
        color="#C23B22",
        linewidth=1.8,
        label="Spline baseline",
    )
    axes[0].set_ylabel("Relative flux [ppm]")
    axes[0].set_title("Input: simulated 10-minute-cadence photometry")

    axes[1].plot(
        result.time_days,
        1e6 * (result.detrended_flux - 1.0),
        ".",
        color="#64818A",
        markersize=1.6,
        alpha=0.65,
        label="Detrended measurements",
    )
    axes[1].plot(
        result.time_days,
        1e6 * (result.model_flux - 1.0),
        color="#C23B22",
        linewidth=1.4,
        label="Injected Ephesos model",
    )
    axes[1].set_ylabel("Relative flux [ppm]")
    axes[1].set_title("Intermediate: transit-masked spline detrending")

    axes[2].plot(
        result.phase,
        1e6 * (result.folded_flux - 1.0),
        ".",
        color="#9AA6A9",
        markersize=1.5,
        alpha=0.35,
        label="Folded measurements",
    )
    axes[2].plot(
        result.phase,
        1e6 * (result.folded_model_flux - 1.0),
        color="#C23B22",
        linewidth=1.5,
        label="Injected Ephesos model",
    )
    axes[2].errorbar(
        result.binned_phase,
        1e6 * (result.binned_flux - 1.0),
        yerr=1e6 * result.binned_uncertainty,
        fmt="o",
        color="black",
        markersize=3.2,
        linewidth=0.8,
        capsize=1.5,
        label="Miletos phase bins",
    )
    axes[2].set_xlim(-0.08, 0.08)
    axes[2].set_xlabel("Orbital phase")
    axes[2].set_ylabel("Relative flux [ppm]")
    axes[2].set_title(f"Output: phase-folded transit, period = {result.period_days:.1f} day")

    for axis in axes[:2]:
        axis.set_xlabel("Time [day]")
    for axis in axes:
        axis.legend(loc="lower right", frameon=True, fancybox=True, framealpha=1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(
        output_path,
        dpi=300 if output_path.suffix == ".png" else None,
        facecolor="white",
    )
    plt.close(figure)
    return output_path


def run_simulated_transit_diagnostic(output_path: Path) -> TransitDiagnostic:
    """Run the deterministic diagnostic and write its pipeline figure."""

    result = analyze_simulated_transit()
    plot_transit_diagnostic(result, output_path)
    return result