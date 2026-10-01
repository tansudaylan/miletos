"""TESS QLP light-curve search, vetting, and injection-recovery tools."""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from astropy.io import fits
from astroquery.mast import Observations
from tdpy.verbosity import print

from .box_least_squares import search_box_least_squares
from .paths import get_data_path


_QLP_NAME = re.compile(r"s(?P<sector>\d{4})-(?P<tic>\d{16})_tess_v\d+_llc\.fits$")


def read_qlp_light_curve(path: str | Path) -> pd.DataFrame:
    """Read one public QLP FITS light curve as normalized BTJD flux."""

    path = Path(path)
    print(f"Reading from {path}...")
    match = _QLP_NAME.search(path.name)
    if match is None:
        raise ValueError(f"Unrecognized QLP light-curve filename: {path.name}")
    with fits.open(path, memmap=True) as hdus:
        table_hdu = next(
            (hdu for hdu in hdus[1:] if getattr(getattr(hdu, "data", None), "names", None)),
            None,
        )
        if table_hdu is None:
            raise ValueError("QLP FITS file has no binary table")
        names = {name.upper(): name for name in table_hdu.data.names}
        flux_name = next((names[name] for name in ("KSPSAP_FLUX", "SAP_FLUX") if name in names), None)
        if "TIME" not in names or flux_name is None:
            raise ValueError("QLP FITS table needs TIME and SAP flux columns")
        error_name = names.get(f"{flux_name.upper()}_ERR")
        if error_name is None:
            raise ValueError(f"QLP FITS table is missing {flux_name}_ERR")
        time = np.asarray(table_hdu.data[names["TIME"]], dtype=float)
        flux = np.asarray(table_hdu.data[flux_name], dtype=float)
        uncertainty = np.asarray(table_hdu.data[error_name], dtype=float)
        quality_name = names.get("QUALITY", names.get("QFLAG"))
        quality = np.zeros(time.size, dtype=int) if quality_name is None else np.asarray(
            table_hdu.data[quality_name]
        )

    if np.nanmedian(time) > 2_000_000.0:
        time = time - 2_457_000.0  # Convert BJD to BTJD [day].
    valid = (
        np.isfinite(time) & np.isfinite(flux) & np.isfinite(uncertainty)
        & (uncertainty > 0) & (quality == 0)
    )
    time, flux, uncertainty = time[valid], flux[valid], uncertainty[valid]
    if time.size < 3:
        raise ValueError("QLP FITS file has fewer than three usable cadences")
    order = np.argsort(time)
    time, flux, uncertainty = time[order], flux[order], uncertainty[order]
    unique = np.concatenate(([True], np.diff(time) > 0.0))
    time, flux, uncertainty = time[unique], flux[unique], uncertainty[unique]
    median_flux = float(np.nanmedian(flux))
    if not np.isfinite(median_flux) or median_flux <= 0.0:
        raise ValueError("QLP flux must have a positive finite median")
    return pd.DataFrame({
        "tic_id": int(match.group("tic")),
        "sector": int(match.group("sector")),
        "time_btjd": time,
        "flux": flux / median_flux,
        "flux_error": uncertainty / median_flux,
    })


def download_qlp_light_curves(
    tic_ids,
    sectors=None,
    cache_directory: str | Path | None = None,
) -> list[Path]:
    """Download FITS light curves from MAST's QLP HLSP for exact TIC IDs."""

    tic_ids = sorted({int(tic_id) for tic_id in tic_ids})
    if not tic_ids or any(tic_id <= 0 for tic_id in tic_ids):
        raise ValueError("tic_ids must contain positive TIC identifiers")
    cache_directory = Path(cache_directory or get_data_path() / "tess_transit_survey" / "mast")
    cache_directory.mkdir(parents=True, exist_ok=True)
    print(f"Querying MAST QLP products for {len(tic_ids)} TIC targets...")
    observations = Observations.query_criteria(
        provenance_name="QLP",
        target_name=[str(tic_id) for tic_id in tic_ids],
        sequence_number=None if sectors is None else [int(sector) for sector in sectors],
    )
    if len(observations) == 0:
        return []
    products = Observations.get_product_list(observations)
    if len(products) == 0:
        return []
    fits_products = products[
        np.char.endswith(np.asarray(products["productFilename"], dtype=str), "_llc.fits")
    ]
    if len(fits_products) == 0:
        return []
    print(f"Writing MAST downloads under {cache_directory}...")
    manifest = Observations.download_products(fits_products, download_dir=str(cache_directory))
    if manifest is None or len(manifest) == 0:
        return []
    return [Path(path) for path in manifest["Local Path"] if path and Path(path).is_file()]


def merge_qlp_light_curves(light_curves: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine sector curves, retaining provenance and unique increasing cadences."""

    if not light_curves:
        raise ValueError("light_curves must not be empty")
    merged = pd.concat(light_curves, ignore_index=True).sort_values("time_btjd")
    merged = merged.drop_duplicates("time_btjd", keep="first").reset_index(drop=True)
    if len(merged) < 3 or np.any(np.diff(merged["time_btjd"]) <= 0.0):
        raise ValueError("combined light curve must contain increasing unique times")
    return merged


def vet_transit_candidate(
    light_curve: pd.DataFrame,
    candidate: dict,
    *,
    snr_threshold: float = 7.0,
    odd_even_threshold: float = 3.0,
    secondary_threshold: float = 5.0,
    inverted_threshold: float = 7.0,
) -> dict[str, float | bool | str]:
    """Measure odd-even, secondary-event, and inverted-curve warning metrics."""

    time = light_curve["time_btjd"].to_numpy(dtype=float)
    flux = light_curve["flux"].to_numpy(dtype=float)
    uncertainty = light_curve["flux_error"].to_numpy(dtype=float)
    period = float(candidate["period_days"])
    duration = float(candidate["duration_days"])
    epoch = float(candidate["transit_time_days"])
    phase = (time - epoch + 0.5 * period) % period - 0.5 * period
    in_transit = np.abs(phase) <= 0.5 * duration
    cycle = np.rint((time - epoch) / period).astype(int)

    def depth_and_error(selection):
        if selection.sum() < 2:
            return np.nan, np.nan
        depth = 1.0 - float(np.median(flux[selection]))
        error = float(np.sqrt(np.sum(uncertainty[selection] ** 2)) / selection.sum())
        return depth, error

    odd_depth, odd_error = depth_and_error(in_transit & (cycle % 2 == 0))
    even_depth, even_error = depth_and_error(in_transit & (cycle % 2 != 0))
    odd_even_snr = (
        abs(odd_depth - even_depth) / np.hypot(odd_error, even_error)
        if np.isfinite(odd_error) and np.isfinite(even_error) and odd_error + even_error > 0.0
        else np.nan
    )
    secondary_phase = (phase - 0.5 * period + 0.5 * period) % period - 0.5 * period
    secondary_depth, secondary_error = depth_and_error(np.abs(secondary_phase) <= 0.5 * duration)
    secondary_snr = (
        secondary_depth / secondary_error
        if np.isfinite(secondary_error) and secondary_error > 0.0 else np.nan
    )

    inverse = search_box_least_squares(
        time,
        2.0 - flux,
        uncertainty,
        duration_days=[duration],
        minimum_period_days=max(0.2, 0.8 * period),
        maximum_period_days=1.2 * period,
    )
    signal_to_noise = float(candidate["depth_snr"])
    flags = {
        "below_snr_threshold": signal_to_noise < snr_threshold,
        "odd_even_mismatch": bool(np.isfinite(odd_even_snr) and odd_even_snr >= odd_even_threshold),
        "secondary_eclipse": bool(np.isfinite(secondary_snr) and secondary_snr >= secondary_threshold),
        "inverted_signal": float(inverse["depth_snr"]) >= inverted_threshold,
    }
    return {
        **candidate,
        "odd_even_snr": float(odd_even_snr),
        "secondary_depth": float(secondary_depth),
        "secondary_snr": float(secondary_snr),
        "inverted_depth_snr": float(inverse["depth_snr"]),
        **flags,
        "disposition": "review" if any(flags.values()) else "candidate",
    }


def search_tess_target(
    light_curve: pd.DataFrame,
    *,
    minimum_period_days: float = 0.5,  # [day]
    maximum_period_days: float = 20.0,  # [day]
    duration_days=(0.04, 0.07, 0.10, 0.14, 0.20),  # [day]
) -> dict:
    """Search one merged target curve with Miletos BLS and return vetting metrics."""

    candidate = search_box_least_squares(
        light_curve["time_btjd"].to_numpy(dtype=float),
        light_curve["flux"].to_numpy(dtype=float),
        light_curve["flux_error"].to_numpy(dtype=float),
        duration_days=duration_days,
        minimum_period_days=minimum_period_days,
        maximum_period_days=maximum_period_days,
    )
    summary = {
        key: candidate[key]
        for key in ("period_days", "transit_time_days", "duration_days", "depth", "depth_uncertainty", "depth_snr")
    }
    return vet_transit_candidate(light_curve, summary)


def injection_recovery(
    light_curve: pd.DataFrame,
    periods_days,
    depths,
    duration_days: float,
    *,
    trials: int = 10,
    seed: int = 0,
    period_tolerance: float = 0.01,
    minimum_depth_snr: float = 7.0,
) -> pd.DataFrame:
    """Measure period-matched, threshold-significant recovery in observed data."""

    periods = np.asarray(periods_days, dtype=float)
    depths = np.asarray(depths, dtype=float)
    if trials < 1 or not np.isfinite(periods).all() or np.any(periods <= duration_days):
        raise ValueError("trials and transit periods must be positive; periods must exceed duration")
    if not np.isfinite(depths).all() or np.any((depths <= 0.0) | (depths >= 1.0)):
        raise ValueError("depths must be finite fractions between zero and one")
    time = light_curve["time_btjd"].to_numpy(dtype=float)
    base_flux = light_curve["flux"].to_numpy(dtype=float)
    uncertainty = light_curve["flux_error"].to_numpy(dtype=float)
    random = np.random.default_rng(seed)
    rows = []
    for period in periods:
        if (time[-1] - time[0]) / period < 3:
            raise ValueError("each injected period must have at least three transits in the baseline")
        for depth in depths:
            recovered = 0
            for _ in range(trials):
                epoch = random.uniform(time[0], time[0] + period)
                phase = (time - epoch + 0.5 * period) % period - 0.5 * period
                injected_flux = base_flux.copy()
                injected_flux[np.abs(phase) < 0.5 * duration_days] *= 1.0 - depth
                result = search_box_least_squares(
                    time,
                    injected_flux,
                    uncertainty,
                    duration_days=[duration_days],
                    minimum_period_days=0.9 * period,
                    maximum_period_days=1.1 * period,
                )
                recovered += (
                    abs(result["period_days"] / period - 1.0) <= period_tolerance
                    and result["depth_snr"] >= minimum_depth_snr
                )
            rows.append({
                "period_days": float(period),
                "depth": float(depth),
                "duration_days": float(duration_days),
                "trials": trials,
                "recoveries": recovered,
                "completeness": recovered / trials,
                "minimum_depth_snr": minimum_depth_snr,
            })
    return pd.DataFrame(rows)