"""Retrieve and normalize public CHEOPS light curves from DACE."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
import re

import numpy as np


def _as_column(data: Mapping, name: str, *, required: bool = True, default=None) -> np.ndarray:
    if name not in data:
        if required:
            raise ValueError(f"DACE CHEOPS response is missing {name!r}.")
        return np.full(len(data.get("file_key", [])), default)
    return np.asarray(data[name])


def _resolve_coordinate_target(Cheops, ra_deg: float, dec_deg: float, radius_arcsec: float) -> tuple[str, list[str]]:
    import astropy.units as u
    from astropy.coordinates import Angle, SkyCoord

    coordinate = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    visits = Cheops.query_region(
        sky_coord=coordinate,
        angle=Angle(radius_arcsec, unit=u.arcsec),
        output_format="dict",
    )
    if not visits or not len(visits.get("file_key", [])):
        raise ValueError("DACE has no CHEOPS visits at the supplied coordinates.")

    names = _as_column(visits, "obj_id_catname").astype(str)
    right_ascension = _as_column(visits, "ra_deg", required=False, default=np.nan).astype(float)
    declination = _as_column(visits, "dec_deg", required=False, default=np.nan).astype(float)
    valid = np.isfinite(right_ascension) & np.isfinite(declination) & (names != "")
    if not valid.any():
        raise ValueError("DACE CHEOPS region results have no target coordinates.")
    separation = coordinate.separation(
        SkyCoord(ra=right_ascension[valid] * u.deg, dec=declination[valid] * u.deg)
    ).arcsec
    nearest_index = np.flatnonzero(valid)[int(np.argmin(separation))]
    if float(np.min(separation)) > radius_arcsec:
        raise ValueError("The nearest CHEOPS target is outside radius_arcsec.")
    target_name = names[nearest_index]
    visit_keys = np.asarray(visits["file_key"]).astype(str)
    target_keys = list(dict.fromkeys(visit_keys[(names == target_name) & valid].tolist()))
    return target_name, target_keys


def _latest_visit_revisions(file_keys: np.ndarray) -> set[str]:
    latest = {}
    for file_key in file_keys:
        match = re.fullmatch(r"(.*)_V(\d{4})", file_key)
        base_key, revision = (match.group(1), int(match.group(2))) if match else (file_key, 0)
        if base_key not in latest or revision > latest[base_key][0]:
            latest[base_key] = revision, file_key
    return {file_key for _, file_key in latest.values()}


def retrieve_cheops_light_curves(
    target: str | int | None = None,
    *,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    radius_arcsec: float = 30.0,
    aperture: str = "default",
    quality_filter: bool = True,
) -> dict[str, np.ndarray]:
    """Return normalized CHEOPS visit curves as ``[BJD, flux, flux_error]`` arrays.

    Supply either a DACE target name/TIC ID or both ICRS coordinates. The optional
    DACE client fetches public archive data without requiring a local API key.
    Each returned value is a distinct visit keyed by its DACE ``file_key``.
    """

    if target is not None and (ra_deg is not None or dec_deg is not None):
        raise ValueError("Supply target or coordinates, not both.")
    if (ra_deg is None) != (dec_deg is None):
        raise ValueError("Both ra_deg and dec_deg are required for coordinate searches.")
    if target is None and ra_deg is None:
        raise ValueError("Supply a target name/TIC ID or ICRS coordinates.")
    if not np.isfinite(radius_arcsec) or radius_arcsec <= 0.0:
        raise ValueError("radius_arcsec must be finite and positive.")
    if aperture.lower() not in {"default", "optimal", "rinf", "rsup"}:
        raise ValueError("aperture must be one of 'default', 'optimal', 'rinf', or 'rsup'.")

    try:
        Cheops = import_module("dace_query.cheops").Cheops
    except ImportError as error:
        raise ImportError("Install CHEOPS support with `pip install 'miletos[cheops]'`.") from error

    filters = None
    if ra_deg is not None:
        target_name, visit_keys = _resolve_coordinate_target(Cheops, float(ra_deg), float(dec_deg), radius_arcsec)
        filters = {"file_key": {"equal": visit_keys}}
    elif isinstance(target, (int, np.integer)):
        target_name = f"TIC {int(target)}"
    else:
        target_name = str(target).strip()
        if not target_name:
            raise ValueError("target must not be empty.")

    data = Cheops.get_lightcurve(
        target_name,
        aperture=aperture.lower(),
        filters=filters,
        output_format="dict",
    )
    if not isinstance(data, Mapping) or not len(data.get("file_key", [])):
        raise ValueError(f"DACE returned no CHEOPS light curves for {target_name!r}.")

    file_keys = _as_column(data, "file_key").astype(str)
    time = _as_column(data, "obj_date_bjd_vect").astype(float)
    if np.nanmedian(time) < 1_000_000.0:
        time = time + 2_400_000.0
    flux = _as_column(data, "photom_flux_vect").astype(float)
    uncertainty = _as_column(data, "photom_flux_vect_err").astype(float)
    if not (file_keys.size == time.size == flux.size == uncertainty.size):
        raise ValueError("DACE CHEOPS columns have inconsistent lengths.")

    latest_visit_keys = _latest_visit_revisions(np.unique(file_keys))
    good = np.isfinite(time) & np.isfinite(flux) & np.isfinite(uncertainty) & (flux > 0.0) & (uncertainty > 0.0)
    if quality_filter:
        for column, maximum in (
            ("photom_event_saa_flag", 0),
            ("photom_event_temp_flag", 0),
            ("photom_event_earth_flag", 1),
            ("photom_event_moon_flag", 0),
            ("photom_event_sun_flag", 0),
            ("photom_event_cosmic_flag", 2),
        ):
            if column in data:
                flag = np.asarray(data[column], dtype=float)
                if flag.shape != time.shape:
                    raise ValueError(f"DACE CHEOPS quality column {column!r} has the wrong length.")
                good &= np.isfinite(flag) & (flag <= maximum)

    curves = {}
    for file_key in dict.fromkeys(file_keys.tolist()):
        if file_key not in latest_visit_keys:
            continue
        selected = good & (file_keys == file_key)
        if selected.sum() < 2:
            continue
        order = np.argsort(time[selected])
        visit_time = time[selected][order]
        visit_flux = flux[selected][order]
        visit_error = uncertainty[selected][order]
        unique = np.concatenate(([True], np.diff(visit_time) > 0.0))
        visit_time, visit_flux, visit_error = visit_time[unique], visit_flux[unique], visit_error[unique]
        median_flux = float(np.median(visit_flux))
        if not np.isfinite(median_flux) or median_flux <= 0.0:
            continue
        curves[file_key] = np.column_stack(
            (visit_time, visit_flux / median_flux, visit_error / median_flux)
        )
    if not curves:
        raise ValueError(f"DACE returned no usable CHEOPS visits for {target_name!r}.")
    return dict(sorted(curves.items(), key=lambda item: item[1][0, 0]))