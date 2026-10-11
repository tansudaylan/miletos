#!/usr/bin/env python3
"""Vet TIC 38455598.1 with Miletos and compare Ephesos exocomet transit models."""

import argparse

import matplotlib

matplotlib.use("agg")

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.constants import G, M_sun, R_sun
from scipy.optimize import least_squares
import ephesos

from miletos.paths import get_data_path
from miletos.tess_transit_search import analyze_tess_target_catalog, plot_target_search
from miletos.tess_transit_survey import read_qlp_light_curve


TIC_ID = 38455598
CANDIDATE = "38455598.1"
SECTORS = (8, 35, 62, 89, 99)
EPHEMERIS_SECTORS = (8, 62, 89)
STELLAR_MASS_SOLAR = 1.18
STELLAR_RADIUS_SOLAR = 1.16325


def _load_sector_curves(cache_directory):
    curves = {}
    for path in Path(cache_directory).rglob("hlsp_qlp_tess_ffi_s*-*_tess_v*_llc.fits"):
        curve = read_qlp_light_curve(path)
        curves[int(curve["sector"].iloc[0])] = curve
    return curves


def _fold_and_bin(curves, results, period_days, epoch_days, bin_width_days=0.004):
    phase_parts = []
    flux_parts = []
    error_parts = []
    sector_parts = []
    for sector in EPHEMERIS_SECTORS:
        curve = curves[sector]
        row = results.loc[results["sector"] == sector].iloc[0]
        time = curve["time_btjd"].to_numpy(dtype=float)
        flux = curve["flux"].to_numpy(dtype=float)
        uncertainty = curve["flux_error"].to_numpy(dtype=float)
        phase = np.remainder(time - epoch_days + 0.5 * period_days, period_days) - 0.5 * period_days
        out_of_transit = np.abs(phase) > 0.15
        baseline = float(np.median(flux[out_of_transit]))
        keep = np.abs(phase) < 0.25
        phase_parts.append(phase[keep])
        flux_parts.append(flux[keep] / baseline)
        error_parts.append(uncertainty[keep] / baseline)
        sector_parts.append(np.full(keep.sum(), sector, dtype=int))

    phase = np.concatenate(phase_parts)
    flux = np.concatenate(flux_parts)
    uncertainty = np.concatenate(error_parts)
    sectors = np.concatenate(sector_parts)
    bin_edges = np.arange(-0.25, 0.25 + bin_width_days, bin_width_days)
    bin_index = np.digitize(phase, bin_edges) - 1
    valid = (bin_index >= 0) & (bin_index < bin_edges.size - 1)
    weights = 1.0 / uncertainty[valid] ** 2
    summed_weights = np.bincount(bin_index[valid], weights=weights, minlength=bin_edges.size - 1)
    summed_flux = np.bincount(
        bin_index[valid], weights=weights * flux[valid], minlength=bin_edges.size - 1
    )
    counts = np.bincount(bin_index[valid], minlength=bin_edges.size - 1)
    populated = counts > 0
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    binned = pd.DataFrame({
        "phase_days": centers[populated],
        "flux": summed_flux[populated] / summed_weights[populated],
        "flux_error": np.sqrt(1.0 / summed_weights[populated]),
        "cadence_count": counts[populated],
    })
    return binned


def _fit_ephesos_models(phase_days, flux, uncertainty, period_days, stellar_mass_solar,
                        stellar_radius_solar):
    semimajor_axis = (
        G * stellar_mass_solar * M_sun * (period_days * 86400.0) ** 2 / (4.0 * np.pi**2)
    ) ** (1.0 / 3.0)
    semimajor_axis_stellar_radii = (semimajor_axis / (stellar_radius_solar * R_sun)).decompose().value

    def disk_model(parameters):
        radius_ratio, impact_parameter = parameters
        return ephesos.evaluate_projected_occultor_model(
            phase_days,
            period_days=period_days,
            equivalent_radius_ratio=radius_ratio,
            summed_radius_to_semimajor_axis=(1.0 + radius_ratio) / semimajor_axis_stellar_radii,
            occultor_type="disk",
            cosine_inclination=impact_parameter / semimajor_axis_stellar_radii,
            limb_darkening_coefficients=(0.35, 0.2),
            grid_size=201,
        )

    disk_fit = least_squares(
        lambda parameters: (disk_model(parameters) - flux) / uncertainty,
        x0=(0.07, 0.3),
        bounds=((0.005, 0.0), (0.25, 0.95)),
        loss="soft_l1",
        max_nfev=40,
    )

    def comet_model(parameters):
        radius_ratio, tail_length, tail_width, tail_tau, tail_decay, impact_parameter = parameters
        return ephesos.evaluate_exocomet_transit_model(
            phase_days,
            period_days=period_days,
            transit_epoch_days=0.0,
            nucleus_radius_ratio=radius_ratio,
            summed_radius_to_semimajor_axis=(1.0 + radius_ratio) / semimajor_axis_stellar_radii,
            tail_length_nucleus_radii=tail_length,
            tail_width_nucleus_radii=tail_width,
            tail_optical_depth=tail_tau,
            tail_decay_length_nucleus_radii=tail_decay,
            tail_widening_per_nucleus_radius=0.02,
            cosine_inclination=impact_parameter / semimajor_axis_stellar_radii,
            limb_darkening_coefficients=(0.35, 0.2),
            grid_size=201,
            mask_grid_size=201,
        )

    comet_fit = least_squares(
        lambda parameters: (comet_model(parameters) - flux) / uncertainty,
        x0=(0.025, 10.0, 0.25, 1.0, 4.0, 0.3),
        bounds=(
            (0.003, 2.0, 0.05, 0.01, 0.5, 0.0),
            (0.2, 30.0, 2.0, 20.0, 20.0, 0.95),
        ),
        loss="soft_l1",
        max_nfev=60,
    )
    disk_flux = disk_model(disk_fit.x)
    comet_flux = comet_model(comet_fit.x)
    disk_chi_square = float(np.sum(((flux - disk_flux) / uncertainty) ** 2))
    comet_chi_square = float(np.sum(((flux - comet_flux) / uncertainty) ** 2))
    number_of_points = flux.size
    return {
        "semimajor_axis_stellar_radii": float(semimajor_axis_stellar_radii),
        "disk_parameters": disk_fit.x,
        "disk_flux": disk_flux,
        "disk_chi_square": disk_chi_square,
        "disk_bic": disk_chi_square + 2 * np.log(number_of_points),
        "comet_parameters": comet_fit.x,
        "comet_flux": comet_flux,
        "comet_chi_square": comet_chi_square,
        "comet_bic": comet_chi_square + 6 * np.log(number_of_points),
    }


def run_analysis(*, download=True):
    root = get_data_path() / f"TIC{TIC_ID}"
    cache_directory = root / "qlp_cache"
    visual_directory = root / "visuals"
    visual_directory.mkdir(parents=True, exist_ok=True)
    targets = pd.DataFrame([
        {
            "tic_id": TIC_ID,
            "sector": sector,
            "tmag": 13.3464,
            "candidate": CANDIDATE,
            "reference_disposition": "Exocomet candidate",
        }
        for sector in SECTORS
    ])
    results, _ = analyze_tess_target_catalog(
        targets,
        visual_directory,
        download=download,
        injection_trials=0,
        data_directory=cache_directory,
    )
    results.to_csv(root / "faintstar_vetting.csv", index=False)
    curves = _load_sector_curves(cache_directory)
    for _, row in results.iterrows():
        sector = int(row["sector"])
        if sector in curves and bool(row["searched"]):
            plot_target_search(
                curves[sector],
                row.to_dict(),
                visual_directory / f"faintstar_sector{sector:02d}.png",
            )

    timing_rows = results.loc[results["sector"].isin(EPHEMERIS_SECTORS)].sort_values("sector")
    timing_epochs = timing_rows["transit_time_days"].to_numpy(dtype=float)
    approximate_period = float(np.median(timing_rows["period_days"].to_numpy(dtype=float)))
    cycle_number = np.rint((timing_epochs - timing_epochs[0]) / approximate_period)
    period_days, epoch_days = np.polyfit(cycle_number, timing_epochs, 1)
    timing_residual = timing_epochs - (epoch_days + period_days * cycle_number)

    folded = _fold_and_bin(curves, results, period_days, epoch_days)
    folded.to_csv(root / "phase_folded_candidate.csv", index=False)
    fit = _fit_ephesos_models(
        folded["phase_days"].to_numpy(),
        folded["flux"].to_numpy(),
        folded["flux_error"].to_numpy(),
        period_days,
        STELLAR_MASS_SOLAR,
        STELLAR_RADIUS_SOLAR,
    )

    phase = folded["phase_days"].to_numpy()
    flux = folded["flux"].to_numpy()
    uncertainty = folded["flux_error"].to_numpy()
    figure, axes = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True,
                                gridspec_kw={"height_ratios": (3, 1)}, constrained_layout=True)
    axes[0].errorbar(phase * 24.0, flux, yerr=uncertainty, fmt=".", markersize=4,
                     color="#555555", alpha=0.65, linewidth=0.5, label="Miletos/QLP fold")
    axes[0].plot(phase * 24.0, fit["disk_flux"], color="#D55E00", linewidth=1.4,
                 label=f"Ephesos opaque disk, BIC={fit['disk_bic']:.1f}")
    axes[0].plot(phase * 24.0, fit["comet_flux"], color="#176B87", linewidth=1.7,
                 label=f"Ephesos dust tail, BIC={fit['comet_bic']:.1f}")
    axes[0].set(ylabel="Relative flux", title=f"TIC {CANDIDATE} | Miletos FaintStar + Ephesos")
    axes[0].legend(frameon=False)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].plot(phase * 24.0, (flux - fit["comet_flux"]) * 1e3, ".", color="#176B87", alpha=0.7)
    axes[1].set(xlabel="Phase from transit center [hour]", ylabel="Tail residual [ppt]")
    figure.savefig(visual_directory / "ephesos_exocomet_model.png", dpi=240, bbox_inches="tight")
    plt.close(figure)

    disk_radius, disk_impact = fit["disk_parameters"]
    nucleus_radius, tail_length, tail_width, tail_tau, tail_decay, tail_impact = fit["comet_parameters"]
    summary = {
        "tic_id": TIC_ID,
        "candidate": CANDIDATE,
        "period_days": float(period_days),
        "epoch_btjd": float(epoch_days),
        "cycle_numbers": ";".join(str(int(value)) for value in cycle_number),
        "epoch_residuals_hours": ";".join(f"{value * 24.0:.3f}" for value in timing_residual),
        "semimajor_axis_stellar_radii": fit["semimajor_axis_stellar_radii"],
        "disk_radius_ratio": float(disk_radius),
        "disk_impact_parameter": float(disk_impact),
        "disk_chi_square": fit["disk_chi_square"],
        "disk_bic": fit["disk_bic"],
        "comet_nucleus_radius_ratio": float(nucleus_radius),
        "comet_tail_length_nucleus_radii": float(tail_length),
        "comet_tail_width_nucleus_radii": float(tail_width),
        "comet_tail_optical_depth": float(tail_tau),
        "comet_tail_decay_length_nucleus_radii": float(tail_decay),
        "comet_impact_parameter": float(tail_impact),
        "comet_chi_square": fit["comet_chi_square"],
        "comet_bic": fit["comet_bic"],
        "model_scope": "circular orbit, opaque nucleus, exponentially fading partially transparent trailing tail",
    }
    pd.DataFrame([summary]).to_csv(root / "ephesos_model_summary.csv", index=False)
    return results, summary, visual_directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="use QLP products already in the Miletos cache")
    arguments = parser.parse_args()
    results, summary, visual_directory = run_analysis(download=not arguments.offline)
    print(results[["sector", "period_days", "depth", "depth_snr", "disposition"]].to_string(index=False))
    print(pd.Series(summary).to_string())
    print(f"Visuals: {visual_directory}")
    return 0


if __name__ == "__main__":
    from tdpy.verbosity import print

    raise SystemExit(main())