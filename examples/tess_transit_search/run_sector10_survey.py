"""Run a random-sample Sector 10 QLP transit survey with Miletos."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from astroquery.mast import Catalogs, Observations
from tdpy.verbosity import print

from miletos.paths import get_data_path, get_repository_path
from miletos.tess_transit_search import (
    analyze_tess_target_catalog,
    estimate_survey_occurrence,
    plot_survey_products,
)


def select_random_qlp_parent_sample(
    *,
    sector: int = 10,
    sample_size: int = 500,
    minimum_tmag: float = 10.5,
    maximum_tmag: float = 13.5,
    seed: int = 20261011,
    catalog_batch_size: int = 1000,
) -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    """Select a reproducible magnitude-limited sample without using search outcomes."""

    if sector < 1 or sample_size < 1 or catalog_batch_size < 1:
        raise ValueError("sector, sample_size, and catalog_batch_size must be positive")
    if not np.isfinite([minimum_tmag, maximum_tmag]).all() or minimum_tmag >= maximum_tmag:
        raise ValueError("TESS magnitude limits must be finite and increasing")

    print(f"Querying public QLP observations for Sector {sector}...")
    observations = Observations.query_criteria(
        provenance_name="QLP",
        sequence_number=sector,
        dataRights="PUBLIC",
    )
    if len(observations) == 0:
        raise ValueError(f"MAST returned no public QLP observations for Sector {sector}")
    tic_ids = pd.to_numeric(observations["target_name"], errors="coerce")
    tic_ids = np.unique(tic_ids.dropna().astype("int64").to_numpy())
    if tic_ids.size < sample_size:
        raise ValueError("The QLP sector catalog is smaller than the requested sample")

    random_order = np.random.default_rng(seed).permutation(tic_ids)
    selected = []
    for start in range(0, random_order.size, catalog_batch_size):
        id_batch = random_order[start : start + catalog_batch_size]
        tic = Catalogs.query_criteria(
            catalog="TIC",
            ID=id_batch.tolist(),
            Tmag=[minimum_tmag, maximum_tmag],
        )
        if len(tic) == 0:
            continue
        matches = tic[["ID", "Tmag", "Teff", "rad", "mass"]].to_pandas()
        matches = matches.drop_duplicates("ID").set_index("ID")
        for tic_id in id_batch:
            if tic_id not in matches.index:
                continue
            row = matches.loc[tic_id]
            tmag = pd.to_numeric(row["Tmag"], errors="coerce")
            if not np.isfinite(tmag):
                continue
            selected.append({
                "tic_id": int(tic_id),
                "sector": int(sector),
                "tmag": float(tmag),
                "teff_kelvin": float(pd.to_numeric(row["Teff"], errors="coerce")),
                "stellar_radius_solar": float(pd.to_numeric(row["rad"], errors="coerce")),
                "stellar_mass_solar": float(pd.to_numeric(row["mass"], errors="coerce")),
            })
            if len(selected) == sample_size:
                break
        if len(selected) == sample_size:
            break

    if len(selected) != sample_size:
        raise ValueError(
            f"Found {len(selected)} TIC matches in the requested magnitude range; "
            f"need {sample_size}"
        )

    targets = pd.DataFrame(selected)
    targets["sample_is_parent_population"] = True
    targets["sample_seed"] = int(seed)
    metadata = {
        "sector": int(sector),
        "public_qlp_observations": int(len(observations)),
        "unique_qlp_tic_ids": int(tic_ids.size),
        "sample_size": int(sample_size),
        "minimum_tmag": float(minimum_tmag),
        "maximum_tmag": float(maximum_tmag),
        "seed": int(seed),
        "selection": "uniform random order over public QLP TIC IDs, then TIC magnitude filter",
    }
    return targets, metadata


def run_sector10_survey(
    *,
    sample_size: int = 500,
    seed: int = 20261011,
    catalog_batch_size: int = 10000,
    injection_trials: int = 3,
    download: bool = True,
    prepare_only: bool = False,
) -> dict[str, Path]:
    """Prepare or analyze a random QLP sample and write search, injection, and rate products."""

    example_path = get_repository_path() / "examples" / "tess_transit_search"
    target_path = example_path / "data" / "sector10_random_parent_sample.csv"
    output_path = example_path / "visuals" / "sector10_random_parent_sample"
    output_path.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        print(f"Reading from {target_path}...")
        targets = pd.read_csv(target_path)
        metadata_path = output_path / "sample_metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    else:
        targets, metadata = select_random_qlp_parent_sample(
            sector=10,
            sample_size=sample_size,
            seed=seed,
            catalog_batch_size=catalog_batch_size,
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing to {target_path}...")
        targets.to_csv(target_path, index=False)
        metadata_path = output_path / "sample_metadata.json"
        print(f"Writing to {metadata_path}...")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    if prepare_only:
        return {"targets": target_path, "metadata": metadata_path}

    results, injections = analyze_tess_target_catalog(
        targets,
        output_path,
        download=download,
        injection_trials=injection_trials,
        seed=seed,
        data_directory=get_data_path() / "tess_transit_survey",
    )
    result_path = output_path / "target_search_results.csv"
    injection_path = output_path / "injection_recovery.csv"
    print(f"Writing to {result_path}...")
    results.to_csv(result_path, index=False)
    print(f"Writing to {injection_path}...")
    injections.to_csv(injection_path, index=False)
    figure_paths = plot_survey_products(results, injections, output_path)

    occurrence_path = output_path / "occurrence.json"
    if not injections.empty:
        try:
            occurrence = {"status": "estimated", **estimate_survey_occurrence(results, injections)}
        except ValueError as error:
            occurrence = {"status": "not_estimated", "reason": str(error)}
    else:
        occurrence = {"status": "not_estimated", "reason": "No injection-recovery results"}
    print(f"Writing to {occurrence_path}...")
    occurrence_path.write_text(json.dumps(occurrence, indent=2) + "\n")

    return {
        "targets": target_path,
        "results": result_path,
        "injections": injection_path,
        "occurrence": occurrence_path,
        **figure_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--catalog-batch-size", type=int, default=10000)
    parser.add_argument("--injection-trials", type=int, default=3)
    parser.add_argument("--offline", action="store_true",
                        help="use QLP FITS files already under the Miletos data directory")
    parser.add_argument("--prepare-only", action="store_true",
                        help="write the random target catalog without downloading or searching")
    arguments = parser.parse_args()
    products = run_sector10_survey(
        sample_size=arguments.sample_size,
        seed=arguments.seed,
        catalog_batch_size=arguments.catalog_batch_size,
        injection_trials=arguments.injection_trials,
        download=not arguments.offline,
        prepare_only=arguments.prepare_only,
    )
    for name, path in products.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
