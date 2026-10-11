"""Search TOI-1233 TESS data for single/repeating transits and TTV-guided moons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import constants as const, units as u
from scipy.ndimage import median_filter

from miletos.transit_search import (
    fit_periodic_ttv,
    search_repeating_transits,
    search_single_transit_events,
    search_transit_origami,
)


BTJD_OFFSET_DAYS = 2457000.0
PLANET_NAMES = ("b", "c", "d", "e")
MILETOS_ROOT = Path(__file__).resolve().parents[2]


def load_toi1233_light_curve(data_directory: str | Path | None = None):
    """Load and sector-normalize the local TOI-1233 Sector 10/11 PDCSAP curves."""

    data_directory = Path(data_directory or MILETOS_ROOT / "data" / "TOI-1233" / "data")
    sector_curves = []
    for sector in (10, 11):
        path = data_directory / f"TESS_PDCSAP_FLUX_Sector{sector}.csv"
        values = np.loadtxt(path, delimiter=",")
        median_flux = float(np.nanmedian(values[:, 1]))
        values[:, 1] /= median_flux
        values[:, 2] /= median_flux
        cadence_days = float(np.nanmedian(np.diff(values[:, 0])))
        window_points = max(31, int(round(0.75 / cadence_days)))
        if window_points % 2 == 0:
            window_points += 1
        baseline = median_filter(values[:, 1], size=window_points, mode="nearest")
        values[:, 1] /= baseline
        values[:, 2] /= baseline
        sector_curves.append((sector, values))

    data = np.concatenate([values for _, values in sector_curves])
    sector_id = np.concatenate([
        np.full(values.shape[0], sector, dtype=int) for sector, values in sector_curves
    ])
    order = np.argsort(data[:, 0])
    data, sector_id = data[order], sector_id[order]
    finite = np.isfinite(data).all(axis=1)
    data, sector_id = data[finite], sector_id[finite]
    unique = np.concatenate(([True], np.diff(data[:, 0]) > 0.0))
    data, sector_id = data[unique], sector_id[unique]
    return data, sector_id, data_directory


def _planet_geometry(posterior: pd.DataFrame, planet_index: int) -> dict[str, float]:
    period_days = float(posterior.loc[f"period__{planet_index}", "mean"])
    radius_ratio = float(posterior.loc[f"r__{planet_index}", "mean"])
    impact_parameter = float(posterior.loc[f"b__{planet_index}", "mean"])
    stellar_mass = float(posterior.loc["m_star", "mean"]) * const.M_sun
    stellar_radius = float(posterior.loc["r_star", "mean"]) * const.R_sun
    semimajor_axis = (
        const.G * stellar_mass * (period_days * u.day) ** 2 / (4.0 * np.pi**2)
    ) ** (1.0 / 3.0)
    scaled_axis = float((semimajor_axis / stellar_radius).decompose().value)
    cosine_inclination = impact_parameter / scaled_axis
    duration_days = period_days / np.pi * np.arcsin(
        np.sqrt((1.0 + radius_ratio) ** 2 - impact_parameter**2)
        / (scaled_axis * np.sqrt(1.0 - cosine_inclination**2))
    )
    return {
        "period_days": period_days,
        "radius_ratio": radius_ratio,
        "impact_parameter": impact_parameter,
        "scaled_semimajor_axis": scaled_axis,
        "duration_days": float(duration_days),
    }


def _ttv_series(posterior: pd.DataFrame, planet_index: int):
    names = [
        f"ttvs_{planet_index}__{epoch}"
        for epoch in range(100)
        if f"ttvs_{planet_index}__{epoch}" in posterior.index
        and f"tts_{planet_index}__{epoch}" in posterior.index
    ]
    timing_days = np.array([
        float(posterior.loc[f"tts_{planet_index}__{name.split('__')[-1]}", "mean"])
        + BTJD_OFFSET_DAYS
        for name in names
    ])
    ttv_days = np.array([float(posterior.loc[name, "mean"]) for name in names])
    uncertainty_days = np.array([float(posterior.loc[name, "sd"]) for name in names])
    return timing_days, ttv_days, uncertainty_days


def _make_intervals(centers, duration_days, padding: float = 1.25):
    return [(float(center - padding * duration_days / 2.0),
             float(center + padding * duration_days / 2.0)) for center in centers]


def run_search(output_directory: str | Path | None = None) -> dict[str, object]:
    """Run repeated, isolated-event, and exploratory ORIGAMI searches offline."""

    output_directory = Path(output_directory or MILETOS_ROOT / "examples" / "TOI-1233"
                            / "visuals" / "transit_search")
    output_directory.mkdir(parents=True, exist_ok=True)
    data, sector_id, data_directory = load_toi1233_light_curve()
    time_days, flux, flux_uncertainty = data.T
    posterior = pd.read_csv(data_directory / "posterior_toi1233_ttv.csv", index_col=0)

    repeating = search_repeating_transits(
        time_days,
        flux,
        flux_uncertainty,
        duration_days=(0.04, 0.07, 0.10, 0.14, 0.20),
        minimum_period_days=0.5,
        maximum_period_days=20.0,
        maximum_candidates=6,
        minimum_depth_snr=7.0,
        minimum_transits=2,
    )
    published_periods = [float(posterior.loc[f"period__{i}", "mean"]) for i in range(4)]
    for candidate in repeating:
        differences = [abs(candidate["period_days"] / period - 1.0) for period in published_periods]
        closest_index = int(np.argmin(differences))
        candidate["closest_published_planet"] = (
            PLANET_NAMES[closest_index] if differences[closest_index] < 0.02 else None
        )

    known_intervals = []
    ttv_data = {}
    planet_durations = {}
    for planet_index in range(3):
        timing_days, ttv_days, uncertainty_days = _ttv_series(posterior, planet_index)
        geometry = _planet_geometry(posterior, planet_index)
        planet_durations[planet_index] = geometry["duration_days"]
        known_intervals.extend(_make_intervals(timing_days, geometry["duration_days"]))
        ttv_data[planet_index] = (timing_days, ttv_days, uncertainty_days, geometry)

    planet_e_candidate = next(
        (candidate for candidate in repeating if candidate["closest_published_planet"] == "e"),
        None,
    )
    if planet_e_candidate is not None:
        period = planet_e_candidate["period_days"]
        epoch = planet_e_candidate["transit_time_days"]
        centers = epoch + np.arange(
            int(np.floor((time_days[0] - epoch) / period)) - 1,
            int(np.ceil((time_days[-1] - epoch) / period)) + 2,
        ) * period
        planet_durations[3] = planet_e_candidate["duration_days"]
        known_intervals.extend(_make_intervals(centers, planet_e_candidate["duration_days"]))

    single_events = search_single_transit_events(
        time_days,
        flux,
        flux_uncertainty,
        duration_days=(0.06, 0.08, 0.10, 0.14, 0.18, 0.24, 0.30),
        baseline_half_window_days=0.5,
        excluded_intervals=known_intervals,
        maximum_candidates=12,
    )

    planet_index = 0
    timing_days, ttv_days, ttv_uncertainty_days, geometry = ttv_data[planet_index]
    epochs = np.arange(timing_days.size, dtype=float)
    maximum_cycle = max(2.1, epochs.size / 2.0)
    ttv_fit = fit_periodic_ttv(
        epochs,
        ttv_days,
        ttv_uncertainty_days,
        minimum_cycle_period=2.0,
        maximum_cycle_period=maximum_cycle,
        grid_size=500,
    )
    planet_c_and_d_intervals = []
    for other_index in (1, 2):
        other_times, _, _, other_geometry = ttv_data[other_index]
        planet_c_and_d_intervals.extend(
            _make_intervals(other_times, other_geometry["duration_days"])
        )
    if planet_e_candidate is not None:
        planet_c_and_d_intervals.extend(
            interval for interval in known_intervals
            if interval not in _make_intervals(timing_days, geometry["duration_days"])
        )

    linear_transit_times = timing_days - ttv_days
    mass_ratio_grid = np.geomspace(1e-3, 1.0, 180)
    origami = search_transit_origami(
        time_days,
        flux,
        flux_uncertainty,
        linear_transit_times_days=linear_transit_times,
        fitted_ttv_days=ttv_fit["fitted_ttv_days"],
        observed_transit_times_days=timing_days,
        transit_duration_days=geometry["duration_days"],
        mass_ratio_grid=mass_ratio_grid,
        excluded_intervals=planet_c_and_d_intervals,
    )
    best_origami = max(origami, key=lambda row: row["delta_chi_squared"]) if origami else None

    pd.DataFrame(repeating).to_csv(output_directory / "repeating_transit_search.csv", index=False)
    pd.DataFrame(single_events).to_csv(output_directory / "single_transit_candidates.csv", index=False)
    pd.DataFrame(origami).to_csv(output_directory / "planet_b_origami_mass_ratio_scan.csv", index=False)

    summary = {
        "target": "HD 108236 (TOI-1233)",
        "data": "TESS PDCSAP Sector 10+11; locally cached; each sector median-normalized and 0.75-day rolling-median detrended",
        "repeating_transits": repeating,
        "repeating_transit_count": len(repeating),
        "single_transit_candidates": single_events,
        "single_transit_snr_threshold": 7.0,
        "single_transit_candidates_above_threshold": sum(
            event["depth_snr"] >= 7.0 for event in single_events
        ),
        "origami": {
            "target_planet": "b",
            "transit_epochs": int(timing_days.size),
            "best_ttv_cycle_period_epochs": ttv_fit["cycle_period"],
            "ttv_amplitude_minutes": ttv_fit["amplitude_minutes"],
            "delta_bic_sinusoid_vs_constant": ttv_fit["delta_bic"],
            "ttv_periodic_support": bool(ttv_fit["delta_bic"] >= 10.0),
            "mass_ratio_grid": [float(mass_ratio_grid[0]), float(mass_ratio_grid[-1])],
            "best_mass_ratio_scan_point": best_origami,
            "interpretation": "Exploratory only; the fitted planet-b TTV sinusoid does not pass Delta BIC >= 10, so the origami fold is not evidence for an exomoon.",
        },
        "caveats": [
            "Single-event SNRs are ranking statistics, not calibrated false-alarm probabilities.",
            "The ORIGAMI result requires a coherent planetary TTV signal and is not interpretable as an exomoon detection when that prerequisite fails.",
            "A full photodynamical fit and null simulations are required for exomoon characterization.",
        ],
    }
    (output_directory / "search_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot_searches(time_days, flux, sector_id, repeating, single_events, output_directory)
    _plot_origami(epochs, ttv_days, ttv_uncertainty_days, ttv_fit, origami, output_directory)
    return summary


def _plot_searches(time_days, flux, sector_id, repeating, single_events, output_directory):
    figure, axes = plt.subplots(3, 1, figsize=(10.0, 9.0), constrained_layout=True)
    for sector in np.unique(sector_id):
        selected = sector_id == sector
        axes[0].scatter(time_days[selected] - BTJD_OFFSET_DAYS, 1e3 * (flux[selected] - 1.0),
                        s=2, alpha=0.45, label=f"Sector {sector}")
    for event in single_events[:5]:
        axes[0].axvline(event["transit_time_days"] - BTJD_OFFSET_DAYS,
                        color="#D18B00", alpha=0.45, linewidth=0.8)
    axes[0].set(xlabel="Time [BTJD day]", ylabel="Relative flux [ppt]",
                title="TOI-1233 TESS data; orange lines mark top isolated-event peaks")
    axes[0].legend(frameon=False)

    labels = [
        f"{candidate.get('closest_published_planet') or 'unmatched'}"
        for candidate in repeating
    ]
    axes[1].bar(np.arange(len(repeating)), [row["depth_snr"] for row in repeating],
                color="#007C78")
    axes[1].set_xticks(np.arange(len(repeating)), labels)
    axes[1].set(xlabel="Closest published planet period", ylabel="BLS depth SNR",
                title="Iterative Miletos repeating-transit search")

    if single_events:
        single_snr = [row["depth_snr"] for row in single_events[:10]]
        single_times = [row["transit_time_days"] - BTJD_OFFSET_DAYS for row in single_events[:10]]
        axes[2].scatter(single_times, single_snr, color="#D18B00", edgecolor="black", linewidth=0.3)
    axes[2].axhline(7.0, color="#A51C30", linestyle="--", linewidth=1.0)
    axes[2].set(xlabel="Single-event epoch [BTJD day]", ylabel="Weighted local-box SNR",
                title="Isolated-event ranking after masking repeated planetary transits")
    figure.savefig(output_directory / "single_and_repeating_transit_search.png",
                   dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_origami(epochs, ttv_days, ttv_uncertainty_days, ttv_fit, origami, output_directory):
    figure, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), constrained_layout=True)
    axes[0].errorbar(epochs, 60.0 * 24.0 * ttv_days,
                     yerr=60.0 * 24.0 * ttv_uncertainty_days,
                     fmt="o", color="#007C78", capsize=2, label="Published TTV posterior means")
    axes[0].plot(epochs, 60.0 * 24.0 * ttv_fit["fitted_ttv_days"],
                 color="#A51C30", label="Best sinusoid")
    axes[0].axhline(0.0, color="#555555", linewidth=0.8)
    axes[0].set(xlabel="Transit epoch", ylabel="Timing residual [min]",
                title=(f"Planet b TTVs: ΔBIC = {ttv_fit['delta_bic']:.2f}; "
                       f"best period = {ttv_fit['cycle_period']:.2f} epochs"))
    axes[0].legend(frameon=False)

    if origami:
        mass_ratio = np.array([row["mass_ratio"] for row in origami])
        delta_chi_squared = np.array([row["delta_chi_squared"] for row in origami])
        axes[1].semilogx(mass_ratio, delta_chi_squared, color="#007C78", linewidth=1.5)
    axes[1].axhline(0.0, color="#555555", linewidth=0.8)
    axes[1].set(xlabel=r"Trial moon-to-planet mass ratio $M_S/M_P$",
                ylabel=r"Signed $\Delta\chi^2$",
                title="TTV-guided transit-origami screening statistic")
    axes[1].text(0.03, 0.96, "Exploratory null scan; no coherent TTV support",
                 transform=axes[1].transAxes, ha="left", va="top")
    figure.savefig(output_directory / "planet_b_transit_origami.png",
                   dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path)
    arguments = parser.parse_args()
    summary = run_search(arguments.output_directory)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
