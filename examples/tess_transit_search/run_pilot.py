"""Run the QLP transit-search and vetting pilot for TOI 1338.01 in Sector 10."""

from pathlib import Path

import pandas as pd
from tdpy.verbosity import print

from miletos.paths import get_data_path, get_repository_path
from miletos.tess_transit_search import (
    analyze_tess_target_catalog,
    plot_survey_products,
    plot_target_search,
)
from miletos.tess_transit_survey import read_qlp_light_curve


PILOT_TARGET = {
    "tic_id": 260128333,
    "toi": "1338.01",
    "sector": 10,
    "tmag": 11.4881,  # [mag]
    "reference_disposition": "PC",
}


def run_pilot(injection_trials: int = 8) -> dict[str, Path]:
    """Search one archived QLP light curve and write reproducible pilot products."""

    example_path = get_repository_path() / "examples" / "tess_transit_search"
    output_path = example_path / "visuals"
    targets = pd.DataFrame([PILOT_TARGET])
    results, injections = analyze_tess_target_catalog(
        targets,
        output_path,
        injection_trials=injection_trials,
        data_directory=get_data_path() / "tess_transit_survey",
    )
    products = plot_survey_products(results, injections, output_path)
    result_path = output_path / "toi1338_sector10_search.csv"
    print(f"Writing to {result_path}...")
    results.to_csv(result_path, index=False)
    injection_path = output_path / "toi1338_sector10_injections.csv"
    print(f"Writing to {injection_path}...")
    injections.to_csv(injection_path, index=False)
    qlp_directory = get_data_path() / "tess_transit_survey"
    qlp_paths = list(qlp_directory.rglob("hlsp_qlp_tess_ffi_s0010-0000000260128333_tess_v*_llc.fits"))
    curve = read_qlp_light_curve(qlp_paths[0])
    plot_target_search(curve, results.iloc[0].to_dict(), output_path / "toi1338_sector10_candidate.png")
    products["target_search"] = output_path / "toi1338_sector10_candidate.png"
    products["results"] = result_path
    products["injections"] = injection_path
    return products


if __name__ == "__main__":
    for name, path in run_pilot().items():
        print(f"{name}: {path}")
