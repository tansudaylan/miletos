"""Run and summarize a reproducible TESS QLP transit-search sample."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tdpy.verbosity import print

from .tess_transit_survey import (
    download_qlp_light_curves,
    injection_recovery,
    merge_qlp_light_curves,
    read_qlp_light_curve,
    search_tess_target,
)
from .paths import get_data_path, get_repository_path


def prepare_toi_target_list(path: str | Path, *, minimum_tmag: float = 10.5,
                            maximum_tmag: float = 13.5, sector: int = 10) -> pd.DataFrame:
    """Select unique QLP-bright TOI targets observed in one requested sector."""

    path = Path(path)
    print(f"Reading from {path}...")
    targets = pd.read_csv(path)
    required = {"TIC ID", "TESS Mag", "Sectors", "TESS Disposition", "TOI"}
    missing = required - set(targets)
    if missing:
        raise ValueError(f"TOI target table is missing columns: {sorted(missing)}")
    sectors = targets["Sectors"].fillna("").astype(str).str.split(",")
    selected = targets[
        pd.to_numeric(targets["TESS Mag"], errors="coerce").between(minimum_tmag, maximum_tmag)
        & sectors.map(lambda values: str(sector) in [value.strip() for value in values])
    ].copy()
    selected["tic_id"] = pd.to_numeric(selected["TIC ID"], errors="coerce")
    selected["tmag"] = pd.to_numeric(selected["TESS Mag"], errors="coerce")
    selected["toi"] = selected["TOI"].astype(str)
    selected["reference_disposition"] = selected["TESS Disposition"].fillna("Unknown").astype(str)
    return (selected.dropna(subset=["tic_id", "tmag"])
            .drop_duplicates("tic_id")[["tic_id", "toi", "tmag", "reference_disposition"]]
            .assign(sector=int(sector), sample_is_parent_population=False).reset_index(drop=True))


def analyze_tess_target_catalog(
    targets: pd.DataFrame,
    output_directory: str | Path,
    *,
    download: bool = True,
    injection_trials: int = 3,
    seed: int = 0,
    data_directory: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Search supplied TIC targets and return per-target and injection tables."""

    required = {"tic_id", "sector"}
    if not required.issubset(targets):
        raise ValueError(f"targets must include columns {sorted(required)}")
    targets = targets.copy()
    targets["tic_id"] = pd.to_numeric(targets["tic_id"], errors="raise").astype("int64")
    targets["sector"] = pd.to_numeric(targets["sector"], errors="raise").astype("int64")
    output_directory = Path(output_directory)
    cache = Path(data_directory or get_data_path() / "tess_transit_survey")
    if download:
        paths = download_qlp_light_curves(
            targets["tic_id"].unique(), targets["sector"].unique(), cache
        )
    else:
        paths = list(cache.rglob("hlsp_qlp_tess_ffi_s*-*_tess_v*_llc.fits"))

    curves = {}
    for path in paths:
        curve = read_qlp_light_curve(path)
        key = (int(curve["tic_id"].iloc[0]), int(curve["sector"].iloc[0]))
        if key in set(zip(targets["tic_id"], targets["sector"])):
            curves.setdefault(key, []).append(curve)

    results = []
    injections = []
    for target_index, target in targets.reset_index(drop=True).iterrows():
        tic_id, sector = int(target["tic_id"]), int(target["sector"])
        result = target.to_dict()
        result.update(searched=False, detection=False, failure="")
        target_curves = curves.get((tic_id, sector), [])
        if not target_curves:
            result["failure"] = "no_qlp_product"
            results.append(result)
            continue
        light_curve = merge_qlp_light_curves(target_curves)
        baseline = float(light_curve["time_btjd"].iloc[-1] - light_curve["time_btjd"].iloc[0])
        result.update(searched=True, cadence_count=len(light_curve), baseline_days=baseline)
        try:
            maximum_period = min(20.0, baseline / 3.0)
            if maximum_period <= 0.5:
                raise ValueError("baseline is too short for three transits")
            candidate = search_tess_target(
                light_curve,
                minimum_period_days=0.5,
                maximum_period_days=maximum_period,
            )
            result.update(candidate)
            result["detection"] = candidate["disposition"] == "candidate"
            if injection_trials > 0:
                periods = np.array([3.0, 10.0])
                periods = periods[periods <= baseline / 3.0]
                if periods.size:
                    recovery = injection_recovery(
                        light_curve,
                        periods,
                        [0.0003, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02],
                        duration_days=0.1,
                        trials=injection_trials,
                        seed=seed + target_index,
                    )
                    recovery["tic_id"] = tic_id
                    recovery["sector"] = sector
                    injections.append(recovery)
        except (ValueError, FloatingPointError) as error:
            result["failure"] = str(error)
        results.append(result)

    return pd.DataFrame(results), (pd.concat(injections, ignore_index=True) if injections else pd.DataFrame())


def estimate_survey_occurrence(
    results: pd.DataFrame,
    injections: pd.DataFrame,
    *,
    injection_grid_weights: pd.Series | None = None,
) -> dict[str, float | int]:
    """Estimate occurrence for a complete target sample and declared injection weighting.

    The weights are indexed by the injection ``(period_days, depth)`` grid. Equal
    weights are used when omitted, corresponding to a uniform discrete planet
    distribution over that grid. This assumption must be appropriate for the
    occurrence domain being reported.
    """

    if "sample_is_parent_population" not in results or not results["sample_is_parent_population"].astype(bool).all():
        raise ValueError("Occurrence rates require a declared complete parent target population")
    searched = results[results["searched"].astype(bool)].copy()
    if searched.empty:
        raise ValueError("No searched targets are available for occurrence estimation")
    if injections.empty:
        raise ValueError("Injection-recovery results are required for occurrence estimation")
    injections = injections.copy()
    grid = pd.MultiIndex.from_frame(injections[["period_days", "depth"]].drop_duplicates())
    if injection_grid_weights is None:
        weights = pd.Series(1.0 / len(grid), index=grid)
    else:
        weights = injection_grid_weights.reindex(grid)
        if weights.isna().any() or (weights < 0).any() or not np.isfinite(weights).all() or weights.sum() <= 0.0:
            raise ValueError("injection_grid_weights must be finite, nonnegative, and cover the injection grid")
        weights /= weights.sum()
    injection_index = pd.MultiIndex.from_frame(injections[["period_days", "depth"]])
    injections["weighted_completeness"] = injections["completeness"].to_numpy() * weights.reindex(
        injection_index
    ).to_numpy()
    per_target = injections.groupby("tic_id")["weighted_completeness"].sum()
    searched["detection_efficiency"] = searched["tic_id"].map(per_target)
    searched = searched.dropna(subset=["detection_efficiency"])
    if searched.empty:
        raise ValueError("No searched targets have measured injection-recovery efficiency")
    try:
        import pergamon
    except ImportError as error:
        raise ImportError("Install Pergamon to calculate occurrence rates") from error
    detections = searched["detection"].astype(int).to_numpy()
    efficiencies = searched["detection_efficiency"].to_numpy(dtype=float)
    occurrence = pergamon.estimate_occurrence_rate(detections, efficiencies)
    from pergamon import log_likelihood_occurrence_rate

    occurrence_grid = np.linspace(0.0, 1.0, 10001)
    log_probability = np.array([
        log_likelihood_occurrence_rate(rate, detections, efficiencies)
        for rate in occurrence_grid
    ])
    probability = np.exp(log_probability - np.max(log_probability))
    probability /= np.trapz(probability, occurrence_grid)
    cumulative = np.concatenate(([0.0], np.cumsum(
        0.5 * (probability[1:] + probability[:-1]) * np.diff(occurrence_grid)
    )))
    credible_interval = np.interp([0.16, 0.5, 0.84], cumulative, occurrence_grid)
    return {
        "searched_targets": int(len(searched)),
        "detections": int(searched["detection"].sum()),
        "maximum_likelihood_occurrence_fraction": occurrence,
        "uniform_prior_posterior_median": float(credible_interval[1]),
        "uniform_prior_68_percent_interval": credible_interval[[0, 2]].tolist(),
        "mean_detection_efficiency": float(searched["detection_efficiency"].mean()),
    }


def plot_survey_products(results: pd.DataFrame, injections: pd.DataFrame,
                         output_directory: str | Path) -> dict[str, Path]:
    """Write separate figures for search yield, completeness, false alarms, and magnitude."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    with plt.rc_context({"font.size": 10, "axes.edgecolor": "black", "axes.facecolor": "white",
                         "figure.facecolor": "white"}):
        figure, axis = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
        counts = [len(results), int(results["searched"].sum()), int(results["detection"].sum())]
        axis.bar(["Catalog targets", "QLP searched", "Candidates"], counts,
                 color=["#8A8F98", "#007C78", "#A51C30"])
        axis.set_ylabel("Number of targets")
        axis.grid(False)
        paths["search_yield"] = output_directory / "tess_transit_search_yield.png"
        print(f"Writing to {paths['search_yield']}...")
        figure.savefig(paths["search_yield"], dpi=300, bbox_inches="tight")
        plt.close(figure)

        if not injections.empty:
            summary = injections.groupby(["period_days", "depth"], as_index=False)["completeness"].mean()
            matrix = summary.pivot(index="depth", columns="period_days", values="completeness")
            figure, axis = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
            image = axis.imshow(matrix.to_numpy(), origin="lower", vmin=0.0, vmax=1.0,
                                cmap="viridis", aspect="auto")
            axis.set_xticks(np.arange(len(matrix.columns)), [f"{value:g}" for value in matrix.columns])
            axis.set_yticks(np.arange(len(matrix.index)), [f"{value * 1e3:g}" for value in matrix.index])
            axis.set_xlabel("Injected orbital period [day]")
            axis.set_ylabel("Injected transit depth [ppt]")
            for row, column in np.ndindex(matrix.shape):
                axis.text(column, row, f"{matrix.iloc[row, column]:.0%}",
                          ha="center", va="center", color="white" if matrix.iloc[row, column] < 0.5 else "black")
            figure.colorbar(image, ax=axis, label="Recovery fraction")
            axis.grid(False)
            paths["completeness"] = output_directory / "tess_transit_completeness.png"
            print(f"Writing to {paths['completeness']}...")
            figure.savefig(paths["completeness"], dpi=300, bbox_inches="tight")
            plt.close(figure)

        searched = results[results["searched"].astype(bool)]
        if not searched.empty and "inverted_depth_snr" in searched:
            figure, axis = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
            axis.hist(searched["inverted_depth_snr"].dropna(), bins=20, color="#64818A",
                      alpha=0.85, label="Inverted-light-curve control")
            axis.axvline(7.0, color="#A51C30", linestyle="--", label="Nominal 7-sigma threshold")
            axis.set(xlabel="Maximum inverted-signal depth SNR", ylabel="Number of targets")
            axis.grid(False)
            axis.legend(frameon=True, fancybox=True, framealpha=1.0)
            paths["false_alarm_control"] = output_directory / "tess_transit_false_alarm_control.png"
            print(f"Writing to {paths['false_alarm_control']}...")
            figure.savefig(paths["false_alarm_control"], dpi=300, bbox_inches="tight")
            plt.close(figure)

        if not searched.empty and "tmag" in searched:
            figure, axis = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
            axis.scatter(searched["tmag"], searched["depth_snr"], s=24, alpha=0.8,
                         color="#007C78", edgecolor="black", linewidth=0.25,
                         label="Searched targets")
            candidates = searched[searched["detection"].astype(bool)]
            if not candidates.empty:
                axis.scatter(candidates["tmag"], candidates["depth_snr"], s=38,
                             color="#A51C30", edgecolor="black", linewidth=0.4,
                             label="Vetted candidates")
            axis.set(xlabel="TESS magnitude [mag]", ylabel="Transit depth SNR")
            axis.grid(False)
            axis.legend(frameon=True, fancybox=True, framealpha=1.0)
            paths["magnitude_yield"] = output_directory / "tess_transit_magnitude_yield.png"
            print(f"Writing to {paths['magnitude_yield']}...")
            figure.savefig(paths["magnitude_yield"], dpi=300, bbox_inches="tight")
            plt.close(figure)

        if not searched.empty and "reference_disposition" in searched:
            reference = searched["reference_disposition"].fillna("Unknown").value_counts()
            figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
            axis.bar(reference.index.astype(str), reference.values, color="#007C78")
            axis.set_ylabel("Searched targets")
            axis.tick_params(axis="x", labelrotation=30)
            axis.grid(False)
            paths["toi_comparison"] = output_directory / "tess_transit_toi_comparison.png"
            print(f"Writing to {paths['toi_comparison']}...")
            figure.savefig(paths["toi_comparison"], dpi=300, bbox_inches="tight")
            plt.close(figure)
    return paths


def plot_target_search(light_curve: pd.DataFrame, result: dict, output_path: str | Path) -> Path:
    """Plot a single observed search and its phase-folded transit candidate."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = light_curve["time_btjd"].to_numpy(dtype=float)
    flux = light_curve["flux"].to_numpy(dtype=float)
    error = light_curve["flux_error"].to_numpy(dtype=float)
    period = float(result["period_days"])
    epoch = float(result["transit_time_days"])
    duration = float(result["duration_days"])
    phase = ((time - epoch + period / 2.0) % period) / period - 0.5
    order = np.argsort(phase)
    phase_model = np.linspace(-0.5, 0.5, 1000)
    model_flux = np.ones_like(phase_model)
    model_flux[np.abs(phase_model) < duration / (2.0 * period)] -= float(result["depth"])
    with plt.rc_context({"font.size": 10, "axes.edgecolor": "black", "axes.facecolor": "white",
                         "figure.facecolor": "white"}):
        figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), constrained_layout=True)
        axes[0].errorbar(time, flux, yerr=error, fmt=".", markersize=2.0,
                         color="#64818A", alpha=0.7, linewidth=0.4)
        axes[0].set(xlabel="Time [BTJD day]", ylabel="Normalized flux", title="Observed QLP light curve")
        axes[1].errorbar(phase[order], flux[order], yerr=error[order], fmt=".", markersize=2.0,
                         color="#64818A", alpha=0.45, linewidth=0.4, label="Observed cadences")
        axes[1].plot(phase_model, model_flux, color="#A51C30", linewidth=1.5,
                     label="BLS box model")
        axes[1].set(xlabel="Orbital phase", ylabel="Normalized flux",
                    title=(f"P = {period:.3f} day, depth = {1e3 * result['depth']:.2f} ppt, "
                           f"SNR = {result['depth_snr']:.1f}, {result['disposition']}"))
        axes[1].legend(frameon=True, fancybox=True, framealpha=1.0)
        for axis in axes:
            axis.grid(False)
        print(f"Writing to {output_path}...")
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True,
                        help="input CSV with tic_id and sector; optional tmag, toi, and reference_disposition")
    parser.add_argument("--output-directory", type=Path, default=(
        get_repository_path() / "examples" / "tess_transit_search" / "visuals"
    ))
    parser.add_argument("--data-directory", type=Path, default=get_data_path() / "tess_transit_survey")
    parser.add_argument("--maximum-targets", type=int)
    parser.add_argument("--injection-trials", type=int, default=3)
    parser.add_argument("--offline", action="store_true",
                        help="use QLP FITS files already under data-directory")
    arguments = parser.parse_args()
    print(f"Reading from {arguments.catalog}...")
    targets = pd.read_csv(arguments.catalog)
    if arguments.maximum_targets is not None:
        if arguments.maximum_targets < 1:
            parser.error("--maximum-targets must be positive")
        targets = targets.head(arguments.maximum_targets)
    results, injections = analyze_tess_target_catalog(
        targets,
        arguments.output_directory,
        download=not arguments.offline,
        injection_trials=arguments.injection_trials,
        data_directory=arguments.data_directory,
    )
    results_path = arguments.output_directory / "tess_transit_search_results.csv"
    injection_path = arguments.output_directory / "tess_transit_injections.csv"
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {results_path}...")
    results.to_csv(results_path, index=False)
    print(f"Writing to {injection_path}...")
    injections.to_csv(injection_path, index=False)
    plot_survey_products(results, injections, arguments.output_directory)
    if (not injections.empty and "sample_is_parent_population" in results
            and results["sample_is_parent_population"].astype(bool).all()):
        summary = estimate_survey_occurrence(results, injections)
        summary_path = arguments.output_directory / "tess_transit_occurrence.json"
        print(f"Writing to {summary_path}...")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())