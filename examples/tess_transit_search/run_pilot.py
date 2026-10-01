"""Run the QLP transit-search and vetting pilot for TOI 1338.01 in Sector 10."""

from pathlib import Path

import pandas as pd
from tdpy.verbosity import print

from miletos.paths import get_data_path, get_repository_path
from miletos.tess_transit_search import (
    analyze_tess_target_catalog,
    prepare_toi_target_list,
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


def run_sector_validation(*, download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    """Search all archived Sector 10 TOIs in the published QLP magnitude range."""

    catalog_path = get_data_path() / "data" / "exofop_toilists_20200916.csv"
    targets_path = get_data_path() / "tess_transit_survey" / "sector10_faintstar_validation_targets.csv"
    targets_path.parent.mkdir(parents=True, exist_ok=True)
    targets = prepare_toi_target_list(catalog_path, minimum_tmag=10.5,
                                      maximum_tmag=13.5, sector=10)  # [mag]
    print(f"Writing to {targets_path}...")
    targets.to_csv(targets_path, index=False)
    visual_path = get_repository_path() / "examples" / "tess_transit_search" / "visuals"
    results, injections = analyze_tess_target_catalog(
        targets, visual_path, download=download, injection_trials=0,
        data_directory=get_data_path() / "tess_transit_survey",
    )
    results_path = visual_path / "sector10_faintstar_validation.csv"
    print(f"Writing to {results_path}...")
    results.to_csv(results_path, index=False)
    products = plot_survey_products(results, injections, visual_path)
    return targets, results, products


if __name__ == "__main__":
    for name, path in run_pilot().items():
        print(f"{name}: {path}")
