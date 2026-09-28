"""Compact, reproducible diagnostics built from the Miletos pipeline APIs."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np

import ephesos
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time
from astropy.utils import iers

from .main import bdtr_tser, fold_tser, rebn_tser

plt.rcParams["text.usetex"] = False


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


@dataclass(frozen=True)
class VisibilityDiagnostic:
    """Nightly and annual visibility for a target and observatory."""

    hours_from_midnight: np.ndarray
    nightly_altitude_degrees: np.ndarray
    nightly_sun_altitude_degrees: np.ndarray
    days_from_year_start: np.ndarray
    annual_max_altitude_degrees: np.ndarray


@dataclass(frozen=True)
class SpectroscopicTransitDiagnostic:
    """Simulated wavelength-resolved transit measurements and recovered depths."""

    time_days: np.ndarray
    wavelength_microns: np.ndarray
    observed_flux: np.ndarray
    model_flux: np.ndarray
    recovered_depth_ppm: np.ndarray
    true_depth_ppm: np.ndarray


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


def analyze_target_visibility(
    right_ascension_degrees: float,
    declination_degrees: float,
    latitude_degrees: float,
    longitude_degrees: float,
    height_meters: float,
    utc_offset_hours: float,
    night: str,
    year_start: str,
) -> VisibilityDiagnostic:
    """Calculate target altitude through one night and across one year."""

    iers.conf.auto_download = False
    target = SkyCoord(right_ascension_degrees * u.deg, declination_degrees * u.deg)
    location = EarthLocation.from_geodetic(
        longitude_degrees * u.deg,
        latitude_degrees * u.deg,
        height_meters * u.m,
    )
    hours_from_midnight = np.linspace(-8.0, 8.0, 193)  # [hour]
    nightly_times = Time(night) + (hours_from_midnight - utc_offset_hours) * u.hour
    nightly_frame = AltAz(obstime=nightly_times, location=location)
    nightly_altitude_degrees = target.transform_to(nightly_frame).alt.deg
    nightly_sun_altitude_degrees = get_sun(nightly_times).transform_to(nightly_frame).alt.deg

    days_from_year_start = np.arange(0.0, 366.0, 7.0)  # [day]
    annual_max_altitude_degrees = np.empty(days_from_year_start.size)  # [deg]
    sample_hours = np.linspace(-8.0, 8.0, 97)  # [hour]
    for index, day in enumerate(days_from_year_start):
        sample_times = (
            Time(year_start)
            + day * u.day
            + (sample_hours - utc_offset_hours) * u.hour
        )
        sample_frame = AltAz(obstime=sample_times, location=location)
        altitude_degrees = target.transform_to(sample_frame).alt.deg
        sun_altitude_degrees = get_sun(sample_times).transform_to(sample_frame).alt.deg
        dark = sun_altitude_degrees < -12.0
        annual_max_altitude_degrees[index] = (
            np.max(altitude_degrees[dark]) if np.any(dark) else np.nan
        )

    return VisibilityDiagnostic(
        hours_from_midnight=hours_from_midnight,
        nightly_altitude_degrees=nightly_altitude_degrees,
        nightly_sun_altitude_degrees=nightly_sun_altitude_degrees,
        days_from_year_start=days_from_year_start,
        annual_max_altitude_degrees=annual_max_altitude_degrees,
    )


def plot_target_visibility(
    result: VisibilityDiagnostic,
    output_path: Path,
    target_label: str,
    observatory_label: str,
) -> Path:
    """Plot nightly altitude and annual dark-time visibility."""

    if output_path.suffix not in {".png", ".pdf"}:
        raise ValueError("output_path must end in .png or .pdf")

    figure, axes = plt.subplots(2, 1, figsize=(8.0, 6.8), constrained_layout=True)
    axes[0].plot(
        result.hours_from_midnight,
        result.nightly_altitude_degrees,
        color="#176B87",
        linewidth=2.2,
        label=target_label,
    )
    axes[0].fill_between(
        result.hours_from_midnight,
        0.0,
        90.0,
        where=result.nightly_sun_altitude_degrees < -12.0,
        color="#DDE7EA",
        label="Astronomical target window",
    )
    axes[0].set_xlabel("Time from local midnight [hour]")
    axes[0].set_ylabel("Altitude [deg]")
    axes[0].set_ylim(0.0, 90.0)
    axes[0].set_title(f"Nightly visibility from {observatory_label}")

    axes[1].plot(
        result.days_from_year_start,
        result.annual_max_altitude_degrees,
        color="#C23B22",
        linewidth=2.2,
        label="Maximum during darkness",
    )
    axes[1].set_xlabel("Time from year start [day]")
    axes[1].set_ylabel("Maximum altitude [deg]")
    axes[1].set_ylim(0.0, 90.0)
    axes[1].set_title("Annual observing accessibility")

    for axis in axes:
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=True, fancybox=True, framealpha=1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(output_path, dpi=300 if output_path.suffix == ".png" else None)
    plt.close(figure)
    return output_path


def run_target_visibility_diagnostic(
    output_path: Path,
    target_label: str,
    observatory_label: str,
    right_ascension_degrees: float,
    declination_degrees: float,
    latitude_degrees: float,
    longitude_degrees: float,
    height_meters: float,
    utc_offset_hours: float,
    night: str,
    year_start: str,
) -> VisibilityDiagnostic:
    """Calculate and plot deterministic target visibility."""

    result = analyze_target_visibility(
        right_ascension_degrees,
        declination_degrees,
        latitude_degrees,
        longitude_degrees,
        height_meters,
        utc_offset_hours,
        night,
        year_start,
    )
    plot_target_visibility(result, output_path, target_label, observatory_label)
    return result


def analyze_simulated_jwst_transit(seed: int = 39) -> SpectroscopicTransitDiagnostic:
    """Simulate a wavelength-resolved transit with broad molecular features."""

    time_days = np.linspace(-0.14, 0.14, 321)  # [day]
    wavelength_microns = np.linspace(0.8, 5.2, 24)  # [micron]
    period_days = 4.0553  # [day]
    radius_ratio = (
        0.145
        + 0.0018 * np.exp(-0.5 * ((wavelength_microns - 1.4) / 0.18) ** 2)
        + 0.0028 * np.exp(-0.5 * ((wavelength_microns - 4.3) / 0.24) ** 2)
    )
    model_flux = np.empty((time_days.size, wavelength_microns.size))
    for index, ratio in enumerate(radius_ratio):
        model_flux[:, index] = ephesos.eval_modl(
            time_days,
            "PlanetarySystem",
            pericomp=np.array([period_days]),
            epocmtracomp=np.array([0.0]),
            rsmacomp=np.array([0.115]),
            cosicomp=np.array([0.0]),
            rratcomp=np.array([ratio]),
            typelmdk="quad",
            booldiag=False,
            typeverb=0,
        )["rflx"][:, 0]

    uncertainty = 350e-6  # [relative flux]
    observed_flux = model_flux + np.random.default_rng(seed).normal(
        0.0,
        uncertainty,
        model_flux.shape,
    )
    in_transit = np.abs(time_days) < 0.045
    out_of_transit = np.abs(time_days) > 0.08
    recovered_depth_ppm = 1e6 * (
        np.mean(observed_flux[out_of_transit], axis=0)
        - np.mean(observed_flux[in_transit], axis=0)
    )
    true_depth_ppm = 1e6 * (
        np.mean(model_flux[out_of_transit], axis=0)
        - np.mean(model_flux[in_transit], axis=0)
    )
    return SpectroscopicTransitDiagnostic(
        time_days=time_days,
        wavelength_microns=wavelength_microns,
        observed_flux=observed_flux,
        model_flux=model_flux,
        recovered_depth_ppm=recovered_depth_ppm,
        true_depth_ppm=true_depth_ppm,
    )


def plot_simulated_jwst_transit(
    result: SpectroscopicTransitDiagnostic,
    output_path: Path,
) -> Path:
    """Plot simulated JWST light curves and their recovered transmission spectrum."""

    if output_path.suffix not in {".png", ".pdf"}:
        raise ValueError("output_path must end in .png or .pdf")

    figure, axes = plt.subplots(2, 1, figsize=(8.0, 7.2), constrained_layout=True)
    selected_indices = np.linspace(
        0,
        result.wavelength_microns.size - 1,
        6,
        dtype=int,
    )
    offset = 0.012
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, selected_indices.size))
    for rank, (index, color) in enumerate(zip(selected_indices, colors)):
        flux_offset = rank * offset
        axes[0].plot(
            result.time_days * 24.0,
            result.observed_flux[:, index] + flux_offset,
            ".",
            color=color,
            markersize=2.0,
            alpha=0.45,
        )
        axes[0].plot(
            result.time_days * 24.0,
            result.model_flux[:, index] + flux_offset,
            color=color,
            linewidth=1.5,
            label=f"{result.wavelength_microns[index]:.1f} micron",
        )
    axes[0].set_xlabel("Time from mid-transit [hour]")
    axes[0].set_ylabel("Relative flux plus offset")
    axes[0].set_title("Explicitly simulated JWST spectrophotometry")

    axes[1].plot(
        result.wavelength_microns,
        result.true_depth_ppm,
        color="#C23B22",
        linewidth=2.0,
        label="Injected spectrum",
    )
    axes[1].plot(
        result.wavelength_microns,
        result.recovered_depth_ppm,
        "o",
        color="#176B87",
        markersize=4.0,
        label="Recovered channel depths",
    )
    axes[1].set_xlabel("Wavelength [micron]")
    axes[1].set_ylabel("Transit depth [ppm]")
    axes[1].set_title("Broad 1.4 and 4.3 micron features are recovered")

    for axis in axes:
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=True, fancybox=True, framealpha=1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(output_path, dpi=300 if output_path.suffix == ".png" else None)
    plt.close(figure)
    return output_path


def run_simulated_jwst_transit_diagnostic(
    output_path: Path,
) -> SpectroscopicTransitDiagnostic:
    """Run and plot the deterministic simulated JWST transit diagnostic."""

    result = analyze_simulated_jwst_transit()
    plot_simulated_jwst_transit(result, output_path)
    return result