#!/usr/bin/env python3
"""Search cached TESS photometry of TIC 266680498 with Miletos BLS vetting."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tdpy.cli import add_plot_arguments
from tdpy.plotting import plot_style
from tdpy.verbosity import print

from miletos.paths import get_data_path
from miletos.tess_transit_search import plot_target_search, search_tess_target


TIC_ID = 266680498
SECTORS = (3, 30, 107)
TIME_OFFSET_DAYS = 2_457_000.0  # [day]
MINIMUM_PERIOD_DAYS = 0.5  # [day]
MAXIMUM_PERIOD_DAYS = 20.0  # [day]
DURATION_DAYS = (0.04, 0.07, 0.10, 0.14, 0.20)  # [day]
DATA_LABEL = "Cached normalized TESS PDCSAP"
VISUAL_DIRECTORY = Path(__file__).resolve().parent / "visuals"


def read_cached_sector_light_curve(path: str | Path, sector: int) -> pd.DataFrame:
    """Read one cached BJD/flux/error CSV as a validated Miletos light curve."""

    path = Path(path)
    print(f"Reading from {path}...")
    table = pd.read_csv(
        path,
        comment="#",
        header=None,
        names=("bjd", "flux", "flux_error"),
    )
    values = table.to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 3:
        raise ValueError("Cached TESS data must contain at least three BJD/flux/error rows")
    if not np.isfinite(values).all() or np.any(values[:, 2] <= 0.0):
        raise ValueError("Cached TESS values must be finite with positive flux errors")
    time_btjd = values[:, 0] - TIME_OFFSET_DAYS
    if np.any(np.diff(time_btjd) <= 0.0):
        raise ValueError("Cached TESS timestamps must be strictly increasing")
    return pd.DataFrame({
        "sector": int(sector),
        "time_btjd": time_btjd,
        "flux": values[:, 1],
        "flux_error": values[:, 2],
    })


def _search_curve(light_curve: pd.DataFrame) -> dict:
    baseline_days = float(
        light_curve["time_btjd"].iloc[-1] - light_curve["time_btjd"].iloc[0]
    )  # [day]
    maximum_period_days = min(MAXIMUM_PERIOD_DAYS, baseline_days / 3.0)  # [day]
    if maximum_period_days <= MINIMUM_PERIOD_DAYS:
        raise ValueError("The light-curve baseline is too short for the requested BLS search")
    result = search_tess_target(
        light_curve,
        minimum_period_days=MINIMUM_PERIOD_DAYS,
        maximum_period_days=maximum_period_days,
        duration_days=DURATION_DAYS,
    )
    return {
        **result,
        "sector": int(light_curve["sector"].iloc[0]) if light_curve["sector"].nunique() == 1 else "combined",
        "cadence_count": len(light_curve),
        "baseline_days": baseline_days,
    }


def run_analysis(
    *,
    data_directory: str | Path | None = None,
    output_directory: str | Path = VISUAL_DIRECTORY,
    typefileplot: str = "png",
    typeplotback: str = "white",
    include_combined: bool = False,
) -> pd.DataFrame:
    """Search each cached sector and the combined observations with Miletos."""

    data_directory = Path(data_directory or get_data_path() / f"TIC{TIC_ID}" / "data")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    curves = {}
    rows = []
    with plot_style(typeplotback):
        for sector in SECTORS:
            input_path = data_directory / f"TESS_PDCSAP_FLUX_Sector{sector:02d}.csv"
            curve = read_cached_sector_light_curve(input_path, sector)
            curves[sector] = curve
            result = _search_curve(curve)
            rows.append(result)
            plot_target_search(
                curve,
                result,
                output_directory
                / f"tic{TIC_ID}_sector{sector:02d}_bls_search.{typefileplot}",
                data_label=DATA_LABEL,
            )

        if include_combined:
            combined = pd.concat(curves.values(), ignore_index=True).sort_values(
                "time_btjd", kind="stable"
            ).reset_index(drop=True)
            combined_result = _search_curve(combined)
            rows.append(combined_result)
            plot_target_search(
                combined,
                combined_result,
                output_directory / f"tic{TIC_ID}_combined_bls_search.{typefileplot}",
                data_label=DATA_LABEL,
            )

    results = pd.DataFrame(rows)
    results_path = data_directory.parent / f"tic{TIC_ID}_miletos_bls_results.csv"
    print(f"Writing to {results_path}...")
    results.to_csv(results_path, index=False)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_arguments(parser)
    parser.add_argument(
        "--typeplotback",
        choices=("white", "dark", "norm", "black"),
        default="white",
    )
    parser.add_argument("--data-directory", type=Path)
    parser.add_argument("--output-directory", type=Path, default=VISUAL_DIRECTORY)
    parser.add_argument(
        "--include-combined",
        action="store_true",
        help="also search the much longer combined multi-sector baseline",
    )
    arguments = parser.parse_args(argv)
    results = run_analysis(
        data_directory=arguments.data_directory,
        output_directory=arguments.output_directory,
        typefileplot=arguments.typefileplot,
        typeplotback=arguments.typeplotback,
        include_combined=arguments.include_combined,
    )
    print(results[["sector", "period_days", "depth", "depth_snr", "disposition"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())