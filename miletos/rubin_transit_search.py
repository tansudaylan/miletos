"""Retrieve and search Rubin Data Preview forced-photometry light curves."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from .transit_search import search_repeating_transits


def _release_schema(release: str) -> str:
    normalized = release.lower()
    if normalized == "dp1":
        return "dp1"
    if normalized in {"dp2", "edp2"}:
        return "dp2"
    raise ValueError("release must be 'dp1' or 'edp2'")


def get_rubin_tap_service(release: str):
    """Get the official TAP client from an authenticated Rubin environment."""

    schema = _release_schema(release)
    if schema == "dp1":
        try:
            from lsst.rsp import get_tap_service
        except ImportError as error:
            raise ImportError(
                "Rubin TAP access requires the lsst.rsp client in an authenticated RSP environment"
            ) from error
        return get_tap_service("tap")

    try:
        from lsst.rsp import RSPDiscovery
    except ImportError as error:
        raise ImportError(
            "Rubin TAP access requires the lsst.rsp client in an authenticated RSP environment"
        ) from error
    return RSPDiscovery("dp2").get_tap_client()


def _as_dataframe(result) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result.copy()
    if hasattr(result, "to_table"):
        result = result.to_table()
    if hasattr(result, "to_pandas"):
        return result.to_pandas()
    return pd.DataFrame(result)


def _table_columns(service, table_name: str) -> dict[str, str]:
    result = _as_dataframe(service.search(
        "SELECT column_name FROM tap_schema.columns "
        f"WHERE table_name = '{table_name}'"
    ))
    if "column_name" not in result:
        raise ValueError(f"TAP schema did not return columns for {table_name}")
    return {str(name).lower(): str(name) for name in result["column_name"]}


def _column(columns: dict[str, str], table_name: str, *choices: str) -> str:
    for choice in choices:
        if choice.lower() in columns:
            return columns[choice.lower()]
    raise ValueError(
        f"{table_name} is missing a required column; tried {', '.join(choices)}"
    )


def _run_async_query(service, query: str) -> pd.DataFrame:
    job = service.submit_job(query)
    try:
        job.run()
        job.wait(phases=["COMPLETED", "ERROR"])
        if job.phase == "ERROR":
            job.raise_if_error()
        if job.phase != "COMPLETED":
            raise RuntimeError(f"Rubin TAP query ended in phase {job.phase!r}")
        return _as_dataframe(job.fetch_result())
    finally:
        if hasattr(job, "delete"):
            job.delete()


def query_rubin_forced_photometry(service, release: str, object_id: int) -> pd.DataFrame:
    """Query one object's forced photometry and its visit metadata via async TAP."""

    schema = _release_schema(release)
    if isinstance(object_id, bool) or not isinstance(object_id, Integral) or object_id <= 0:
        raise ValueError("object_id must be a positive integer")

    forced_table = f"{schema}.ForcedSource"
    visit_table = f"{schema}.Visit"
    forced = _table_columns(service, forced_table)
    visits = _table_columns(service, visit_table)
    forced_object = _column(forced, forced_table, "objectId")
    forced_visit = _column(forced, forced_table, "visit", "visitId")
    flux = _column(forced, forced_table, "psfFlux")
    flux_error = _column(forced, forced_table, "psfFluxErr")
    flux_flag = _column(forced, forced_table, "psfFlux_flag")
    visit_id = _column(visits, visit_table, "visitId", "visit")
    band = _column(visits, visit_table, "band")

    midpoint = next(
        (visits[name.lower()] for name in (
            "expMidptMJD", "observationMidpointMJD", "observation_midpoint_mjd"
        ) if name.lower() in visits),
        None,
    )
    exposure = _column(visits, visit_table, "expTime", "exposureTime", "exposure_time")
    if midpoint is not None:
        time_expression = f"v.{midpoint} AS time_mjd"
    else:
        start = _column(
            visits, visit_table, "obsStartMJD", "observationStartMJD", "observation_start_mjd"
        )
        time_expression = (
            f"(v.{start} + v.{exposure} / 172800.0) AS time_mjd"
        )

    selected_flag = f", fs.{flux_flag} AS psf_flux_flag" if flux_flag else ""
    query = f"""SELECT
    fs.{forced_object} AS object_id,
    v.{visit_id} AS visit_id,
    v.{band} AS band,
    {time_expression},
    v.{exposure} AS exposure_time_s,
    fs.{flux} AS psf_flux_njy,
    fs.{flux_error} AS psf_flux_error_njy{selected_flag}
FROM {forced_table} AS fs
JOIN {visit_table} AS v ON fs.{forced_visit} = v.{visit_id}
WHERE fs.{forced_object} = {int(object_id)}
ORDER BY v.{visit_id}"""
    result = _run_async_query(service, query)
    if not result.empty and not (result["object_id"] == int(object_id)).all():
        raise ValueError("Rubin TAP returned rows for a different object_id")
    return result


def prepare_rubin_light_curve(
    rows: pd.DataFrame,
    *,
    ra_deg: float,
    dec_deg: float,
    observatory_location: EarthLocation,
    time_scale: str,
) -> pd.DataFrame:
    """Convert queried nJy fluxes and visit MJD to per-band relative BJD_TDB."""

    required = {
        "object_id", "visit_id", "band", "time_mjd", "psf_flux_njy",
        "psf_flux_error_njy",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Rubin photometry is missing columns: {sorted(missing)}")
    if not isinstance(observatory_location, EarthLocation):
        raise TypeError("observatory_location must be an astropy EarthLocation")
    if not np.isfinite([ra_deg, dec_deg]).all() or not -90.0 <= dec_deg <= 90.0:
        raise ValueError("target coordinates must be finite with declination in [-90, 90]")

    photometry = rows.copy()
    for name in ("time_mjd", "psf_flux_njy", "psf_flux_error_njy"):
        photometry[name] = pd.to_numeric(photometry[name], errors="coerce")
    valid = (
        np.isfinite(photometry[["time_mjd", "psf_flux_njy", "psf_flux_error_njy"]]).all(axis=1)
        & (photometry["psf_flux_error_njy"] > 0.0)
    )
    if "psf_flux_flag" in photometry:
        valid &= ~photometry["psf_flux_flag"].fillna(True).astype(bool)
    photometry = photometry.loc[valid].copy()
    if photometry.empty:
        raise ValueError("Rubin photometry has no usable forced-flux measurements")

    coordinates = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    observed_time = Time(
        photometry["time_mjd"].to_numpy(dtype=float),
        format="mjd",
        scale=time_scale,
        location=observatory_location,
    )
    photometry["time_bjd_tdb"] = (
        observed_time.tdb + observed_time.light_travel_time(coordinates, kind="barycentric")
    ).value

    photometry["band"] = photometry["band"].astype(str)
    photometry["flux"] = np.nan
    photometry["flux_error"] = np.nan
    for _, indices in photometry.groupby("band").groups.items():
        band_flux = photometry.loc[indices, "psf_flux_njy"].to_numpy(dtype=float)
        median_flux = float(np.median(band_flux))
        if median_flux <= 0.0:
            photometry.loc[indices, "psf_flux_error_njy"] = np.nan
            continue
        photometry.loc[indices, "flux"] = band_flux / median_flux
        photometry.loc[indices, "flux_error"] = (
            photometry.loc[indices, "psf_flux_error_njy"].to_numpy(dtype=float) / median_flux
        )
    photometry = photometry.dropna(subset=["flux", "flux_error"])
    if photometry.empty:
        raise ValueError("No Rubin band has a positive median flux")
    return photometry.sort_values(["time_bjd_tdb", "band"]).reset_index(drop=True)


def search_rubin_target(
    light_curve: pd.DataFrame,
    *,
    minimum_period_days: float = 0.5,
    maximum_period_days: float = 20.0,
    duration_days: Iterable[float] = (0.04, 0.07, 0.10, 0.14, 0.20),
    minimum_depth_snr: float = 7.0,
) -> list[dict[str, float | int | str]]:
    """Rank repeating-transit BLS peaks independently in each Rubin band."""

    required = {"time_bjd_tdb", "flux", "flux_error", "band"}
    missing = required - set(light_curve.columns)
    if missing:
        raise ValueError(f"Rubin light curve is missing columns: {sorted(missing)}")
    duration_grid = tuple(duration_days)
    candidates = []
    for band, samples in light_curve.groupby("band"):
        samples = samples.sort_values("time_bjd_tdb").copy()
        samples["weight"] = 1.0 / samples["flux_error"].to_numpy(dtype=float) ** 2
        samples["weighted_flux"] = samples["weight"] * samples["flux"].to_numpy(dtype=float)
        binned = samples.groupby("time_bjd_tdb", as_index=False).agg(
            weight=("weight", "sum"), weighted_flux=("weighted_flux", "sum")
        )
        if len(binned) < 3:
            continue
        times = binned["time_bjd_tdb"].to_numpy(dtype=float)
        baseline_days = float(times[-1] - times[0])
        maximum_search_period = min(maximum_period_days, baseline_days / 3.0)
        if maximum_search_period <= minimum_period_days:
            continue
        flux = binned["weighted_flux"].to_numpy(dtype=float) / binned["weight"].to_numpy(dtype=float)
        uncertainty = np.sqrt(1.0 / binned["weight"].to_numpy(dtype=float))
        for candidate in search_repeating_transits(
            times,
            flux,
            uncertainty,
            duration_days=duration_grid,
            minimum_period_days=minimum_period_days,
            maximum_period_days=maximum_search_period,
            minimum_depth_snr=minimum_depth_snr,
            minimum_transits=3,
        ):
            candidates.append({
                **candidate,
                "band": str(band),
                "baseline_days": baseline_days,
                "number_of_visits": int(len(binned)),
                "disposition": "unvetted",
            })
    return candidates