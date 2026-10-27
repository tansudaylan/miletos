"""Transit measurements shared with the Albus white-dwarf searches."""

import numpy as np
from scipy.signal import savgol_filter


def period_at_max_power(bls_periodogram):
    """Return the fitted period without changing its units."""
    return bls_periodogram.period_at_max_power


def duration_at_max_power(bls_periodogram):
    """Return the fitted duration without changing its units."""
    return bls_periodogram.duration_at_max_power


def transit_depth(detrended_time, detrended_flux, transit_mask):
    """Return median, mean, and maximum depths relative to unit baseline."""
    in_transit_flux = detrended_flux[transit_mask]
    if in_transit_flux.size == 0:
        return np.nan, np.nan, np.nan
    return (1 - np.median(in_transit_flux), 1 - np.mean(in_transit_flux),
            1 - np.min(in_transit_flux))


def transit_v_shape(detrended_time, detrended_flux, transit_mask):
    """Retain Albus's threshold-based shape statistic on the supplied samples.

    This uses the span of the selected times, without separating repeated
    events. It is a screening statistic rather than a fitted ingress fraction.
    """
    time_in_transit = detrended_time[transit_mask]
    flux_in_transit = detrended_flux[transit_mask]
    depth = 1 - np.median(flux_in_transit)
    flux_10 = 1 - 0.1 * depth
    flux_90 = 1 - 0.9 * depth
    ingress_candidates = np.where(flux_in_transit > flux_10)[0]
    egress_candidates = np.where(flux_in_transit > flux_90)[0]
    if len(ingress_candidates) == 0 or len(egress_candidates) == 0:
        return np.nan
    t_ingress = time_in_transit[ingress_candidates[-1]] - time_in_transit[0]
    t_egress = time_in_transit[-1] - time_in_transit[egress_candidates[0]]
    total_duration = time_in_transit[-1] - time_in_transit[0]
    return (t_ingress + t_egress) / total_duration


def out_of_transit_variability(detrended_flux, transit_mask):
    """Return population scatter outside transit, or NaN without samples."""
    out_of_transit_flux = detrended_flux[~transit_mask]
    if out_of_transit_flux.size == 0:
        return np.nan
    return np.std(out_of_transit_flux)


def transit_snr(detrended_flux, transit_mask):
    """Return mean depth divided by out-of-transit population scatter.

    No square-root sample-count factor is applied. This is distinct from the
    fitted depth SNR returned by a BLS search.
    """
    in_transit_flux = detrended_flux[transit_mask]
    if in_transit_flux.size == 0:
        return np.nan
    depth = 1 - np.mean(in_transit_flux)
    noise = out_of_transit_variability(detrended_flux, transit_mask)
    return depth / noise if noise > 0 else np.nan


def _sigma_clip(x, sigma=3.0, max_iter=3):
    x = np.asarray(x, dtype=float)
    mask = np.ones(x.size, dtype=bool)
    for _ in range(max_iter):
        xm = np.median(x[mask])
        xs = 1.4826 * np.median(np.abs(x[mask] - xm))
        if xs == 0:
            break
        new_mask = np.abs(x - xm) <= sigma * xs
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask
    return x[mask]


def _wrap_centered_phase(phase_array, center=0.0):
    """Center phases and wrap to [-0.5, 0.5)."""
    phase = np.asarray(getattr(phase_array, 'value', phase_array), dtype=float)
    center = float(center) if hasattr(center, '__float__') else np.asarray(center, dtype=float)
    return (phase - center + 0.5) % 1.0 - 0.5


def transit_depth_quantile_phase(
    lightcurve, duration, phase0=0.0, nbins=150, tau=0.20, guard_factor=1.2,
    clip_sigma=3.0, smooth_window=11, smooth_poly=3, window_scale=1.5,
):
    """Measure a smoothed lower-quantile depth in period-normalized phase.

    Inputs and outputs retain Albus's conventions. ``duration`` is a phase
    width, not a time in days. The returned depth error uses the clipped
    out-of-transit MAD scatter and the quantile-weighted support count. It
    does not include detrending or correlated-noise uncertainty.
    """
    phase = np.asarray(lightcurve['time'])
    flux = np.asarray(lightcurve['flux'])
    centered_phase = _wrap_centered_phase(phase, center=phase0)
    half_duration = 0.5 * duration
    oot_mask = np.abs(centered_phase) > guard_factor * half_duration
    if oot_mask.sum() < 20:
        raise ValueError('Too few OOT points; reduce guard_factor or check duration/phase0.')
    oot_flux_clipped = _sigma_clip(flux[oot_mask], sigma=clip_sigma)
    baseline = float(np.median(oot_flux_clipped))
    sigma_oot = float(1.4826 * np.median(np.abs(oot_flux_clipped - baseline)))
    window_halfwidth = window_scale * half_duration
    window_mask = np.abs(centered_phase) <= window_halfwidth
    window_phase = centered_phase[window_mask]
    window_flux = flux[window_mask]
    if window_phase.size < max(50, nbins):
        raise ValueError('Not enough points in the transit window for the chosen nbins.')
    edges = np.linspace(-window_halfwidth, window_halfwidth, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    quantiles = np.full(nbins, np.nan)
    counts = np.zeros(nbins, dtype=int)
    for index in range(nbins):
        selection = (window_phase >= edges[index]) & (window_phase < edges[index + 1])
        if not np.any(selection):
            continue
        samples = _sigma_clip(window_flux[selection], sigma=clip_sigma)
        if samples.size == 0:
            continue
        quantiles[index] = np.quantile(samples, tau)
        counts[index] = samples.size
    populated = ~np.isnan(quantiles)
    if populated.sum() < 7:
        raise ValueError('Too few populated bins\u2014reduce nbins or increase window_scale.')
    centers = centers[populated]
    quantiles = quantiles[populated]
    width = min(smooth_window, populated.sum() if populated.sum() % 2 == 1 else populated.sum() - 1)
    width = max(5, width)
    if width % 2 == 0:
        width -= 1
    smooth_poly = min(smooth_poly, width - 1)
    smoothed = savgol_filter(quantiles, width, smooth_poly)
    floor_index = int(np.argmin(smoothed))
    floor = float(smoothed[floor_index])
    floor_phase = float(centers[floor_index]) + phase0
    depth = float(baseline - floor)
    support_bins = smoothed <= floor + 0.25 * depth
    effective_count = max(1.0, float(counts[populated][support_bins].sum()) * tau)
    snr = depth / (sigma_oot / np.sqrt(effective_count))
    depth_err = sigma_oot / np.sqrt(effective_count)
    return {
        'depth': depth, 'depth_err': float(depth_err), 'floor': floor,
        'floor_phase': floor_phase, 'F0': baseline, 'sigma_oot': sigma_oot,
        'snr': float(snr), 'N_eff': float(effective_count),
        'bin_centers': centers + phase0, 'bin_quantile': quantiles,
        'bin_quantile_sm': smoothed, 'window_mask': window_mask, 'oot_mask': oot_mask,
    }