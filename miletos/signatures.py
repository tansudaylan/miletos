"""Compact-object companion signatures and derived physical features."""

import numpy as np
import ephesos
import tdpy


compute_photometric_signatures = ephesos.evaluate_compact_object_signatures


def derive_compact_object_features(
    stellar_radius_solar, period_days, companion_mass_solar, stellar_mass_solar
) -> dict[str, np.ndarray]:
    """Return Ephesos photometric amplitudes and derived orbital features."""

    if any(tdpy.numerics.has_masked_or_complex_values(value) for value in (
        stellar_radius_solar, period_days, companion_mass_solar, stellar_mass_solar
    )):
        raise ValueError("Compact-object inputs must contain measured real values")
    period_days, stellar_radius_solar, companion_mass_solar, stellar_mass_solar = (
        np.atleast_1d(value)
        for value in np.broadcast_arrays(
            *(
                np.asarray(value, dtype=float)
                for value in (
                    period_days,
                    stellar_radius_solar,
                    companion_mass_solar,
                    stellar_mass_solar,
                )
            )
        )
    )
    if any(not np.isfinite(value).all() or np.any(value <= 0.0) for value in (
        period_days, stellar_radius_solar, companion_mass_solar, stellar_mass_solar
    )):
        raise ValueError("Compact-object inputs must be finite and positive")
    signatures = compute_photometric_signatures(
        period_days,
        companion_mass_solar,
        stellar_radius_solar=stellar_radius_solar,
        stellar_mass_solar=stellar_mass_solar,
        stellar_density_cgs=tdpy.stellar_bulk_density(
            stellar_mass_solar, stellar_radius_solar, solar_density_g_cm3=1.41
        ),  # [g cm^-3]
    )
    semimajor_axis_solar = tdpy.retr_smaxkepl(
        period_days, stellar_mass_solar + companion_mass_solar
    ) * 215.0  # [R_Sun]
    duration_hours = tdpy.retr_duratrantotl(
        period_days,
        stellar_radius_solar / semimajor_axis_solar,
        np.zeros_like(period_days),
    )  # [hour]
    schwarzschild_radius_solar = 4.24e-6 * companion_mass_solar  # [R_Sun]
    return {
        "amplbeammodl": np.atleast_1d(signatures["beaming"]),
        "ampellimodl": np.atleast_1d(signatures["ellipsoidal"]),
        "amplslenmodl": np.atleast_1d(signatures["self_lensing"]),
        "duratrantotlmodl": np.atleast_1d(duration_hours),
        "smaxmodl": np.atleast_1d(semimajor_axis_solar),
        "radischw": np.atleast_1d(schwarzschild_radius_solar),
    }