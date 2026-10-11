"""Optional physical and structural checks for the Miletos pipeline."""

import numpy as np


def check_time_series_sanity(time_series, *, boolsanity=False, label="time series"):
    """Check one standardized (observation, channel, time/flux/error) array."""

    if not boolsanity:
        return
    values = np.asarray(time_series, dtype=float)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 1 or values.shape[2] != 3:
        raise ValueError(f"{label} must have shape (observation, channel, 3)")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains nonfinite time, flux, or uncertainty values")
    if np.any(values[..., 2] < 0.0):
        raise ValueError(f"{label} contains negative flux uncertainties")
    if np.any(np.diff(values[:, 0, 0]) < 0.0):
        raise ValueError(f"{label} timestamps must be sorted")


def check_transit_parameter_sanity(
    period_days,
    epoch_days,
    scaled_semimajor_axis,
    cosine_inclination,
    duration_hours,
    eccentricity_cosine,
    eccentricity_sine,
    *,
    boolsanity=False,
):
    """Check companion ephemerides and orbital geometry for physical consistency."""

    if not boolsanity:
        return
    vectors = {
        "period": np.asarray(period_days, dtype=float),
        "epoch": np.asarray(epoch_days, dtype=float),
        "scaled semimajor axis": np.asarray(scaled_semimajor_axis, dtype=float),
        "cosine inclination": np.asarray(cosine_inclination, dtype=float),
        "duration": np.asarray(duration_hours, dtype=float),
        "eccentricity cosine": np.asarray(eccentricity_cosine, dtype=float),
        "eccentricity sine": np.asarray(eccentricity_sine, dtype=float),
    }
    if any(values.ndim != 1 for values in vectors.values()):
        raise ValueError("companion parameters must be one-dimensional vectors")
    if len({values.size for values in vectors.values()}) != 1:
        raise ValueError("companion parameter vectors must have matching lengths")

    for name in (
        "period", "epoch", "scaled semimajor axis", "cosine inclination",
        "eccentricity cosine", "eccentricity sine",
    ):
        if not np.isfinite(vectors[name]).all():
            raise ValueError(f"companion {name} values must be finite")
    if np.any(vectors["period"] <= 0.0):
        raise ValueError("companion periods must be positive")
    if np.any(vectors["scaled semimajor axis"] <= 0.0):
        raise ValueError("scaled semimajor axes must be positive")
    if np.any(np.abs(vectors["cosine inclination"]) > 1.0):
        raise ValueError("cosine inclinations must lie in [-1, 1]")
    eccentricity = np.hypot(vectors["eccentricity cosine"], vectors["eccentricity sine"])
    if np.any(eccentricity >= 1.0):
        raise ValueError("bound companion orbits require eccentricity below one")

    durations = vectors["duration"]
    if np.isinf(durations).any():
        raise ValueError("finite transit durations must be positive and shorter than the period")
    finite_duration = np.isfinite(durations)
    if np.any(durations[finite_duration] <= 0.0):
        raise ValueError("finite transit durations must be positive")
    if np.any(durations[finite_duration] >= 24.0 * vectors["period"][finite_duration]):
        raise ValueError("finite transit durations must be shorter than the orbital period")


def check_model_flux_sanity(model_flux, time_days, *, label="model flux", boolsanity=False):
    """Check that modeled photometric flux is finite, positive, and time-aligned."""

    if not boolsanity:
        return
    model = np.asarray(model_flux, dtype=float)
    time = np.asarray(time_days, dtype=float)
    if time.ndim != 1 or model.ndim not in {1, 2} or model.shape[0] != time.size:
        raise ValueError(f"{label} must have one model value per observation time")
    if not np.isfinite(time).all() or not np.isfinite(model).all():
        raise ValueError(f"{label} and its observation times must be finite")
    if np.any(model <= 0.0):
        raise ValueError(f"{label} must remain positive")


def check_phase_fold_sanity(phase_series, *, phase_center=0.5, label="phase fold", boolsanity=False):
    """Check sorted folded samples lie within the cycle implied by phase_center."""

    if not boolsanity:
        return
    values = np.asarray(phase_series, dtype=float)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 1 or values.shape[2] != 3:
        raise ValueError(f"{label} must have shape (observation, channel, 3)")
    if not np.isfinite(values).all() or not np.isfinite(phase_center):
        raise ValueError(f"{label} and phase_center must be finite")
    phase = values[:, 0, 0]
    lower, upper = phase_center - 0.5, phase_center + 0.5
    tolerance = 1e-12
    if np.any(np.diff(phase) < 0.0):
        raise ValueError(f"{label} phase values must be sorted")
    if np.any(phase < lower - tolerance) or np.any(phase >= upper + tolerance):
        raise ValueError(f"{label} values must lie in [{lower}, {upper})")