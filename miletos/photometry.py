"""Native TESS preprocessing and uncertainty-bearing image-stack photometry."""

import numpy as np
from tdpy.numerics import has_masked_or_complex_values
from .light_curve import LightCurve, as_light_curve


class TessDownloadError(RuntimeError):
    """MAST product retrieval failed after a successful query."""


def download_tess_light_curves(target, *, tic_id=True):
    """Retrieve TESS light-curve FITS products directly through MAST."""
    from astroquery.mast import Observations

    criteria = dict(obs_collection='TESS', dataproduct_type='timeseries')
    if tic_id:
        criteria['target_name'] = str(target).removeprefix('TIC ').strip()
    else:
        criteria.update(objectname=str(target), radius='0.001 deg')
    observations = Observations.query_criteria(**criteria)
    if len(observations) == 0:
        return None
    products = Observations.get_product_list(observations)
    names = np.asarray(products['productFilename'], dtype=str)
    selected = products[np.char.endswith(names, '_lc.fits') | np.char.endswith(names, '_llc.fits')]
    if len(selected) == 0:
        return None
    from tdpy.verbosity import print

    print('Writing to mastDownload...')
    try:
        manifest = Observations.download_products(selected, download_dir='.', mrp_only=False)
    except Exception as error:
        raise TessDownloadError('MAST light-curve download failed') from error
    if manifest is None or len(manifest) == 0:
        return None
    return [read_tess_light_curve(filename) for filename, status in
            zip(manifest['Local Path'], manifest['Status']) if status == 'COMPLETE'] or None


def read_tess_light_curve(filename, *, quality_bitmask=175):
    """Read SPOC or QLP flux and errors with the standard TESS quality mask."""
    from astropy.io import fits
    from astropy.time import Time
    from tdpy.verbosity import print

    print(f'Reading from {filename}...')
    with fits.open(filename, memmap=False) as hdus:
        table = hdus[1].data
        names = set(table.names)
        pair = next(((name, name + '_ERR') for name in
                     ('PDCSAP_FLUX', 'KSPSAP_FLUX', 'DET_FLUX', 'SAP_FLUX')
                     if name in names), None)
        if pair is None or 'TIME' not in names:
            raise ValueError('TESS FITS requires TIME and a supported flux column')
        flux_name, error_name = pair
        time = np.array(table['TIME'], dtype=float)
        flux = np.array(table[flux_name], dtype=float)
        error = np.array(table[error_name], dtype=float) if error_name in names else np.full_like(flux, np.nan)
        quality_name = 'QUALITY' if 'QUALITY' in names else 'QFLAG' if 'QFLAG' in names else None
        quality = np.zeros(len(time), dtype=int) if quality_name is None else np.asarray(table[quality_name], dtype=int)
        keep = np.isfinite(time) & ((quality & quality_bitmask) == 0)
        reference = hdus[1].header.get('BJDREFI', hdus[0].header.get('BJDREFI', 2457000))
        reference += hdus[1].header.get('BJDREFF', hdus[0].header.get('BJDREFF', 0.))
        time = Time(time[keep] + reference, format='jd', scale='tdb')
        return LightCurve(time, flux[keep], error[keep], dict(hdus[0].header))


def preprocess_tess_light_curve(input, TICID=True, injection=False, plots=False, report=False):
    """Download native TESS curves and apply Albus's median-filter convention."""
    from tdpy.verbosity import print

    try:
        collection = download_tess_light_curves(input, tic_id=TICID)
    except TessDownloadError:
        print('download failed')
        return None
    if collection is None:
        print('download empty')
        return None
    return prepare_light_curve_collection(
        collection, target_label=input, injection=injection, plots=plots, report=report,
    )


def prepare_light_curve_collection(light_curves, *, target_label='', injection=False,
                                   plots=False, report=False):
    """Normalize, stitch, and median-detrend curves using Albus's convention.

    Injection baselines are centered on zero, other curves on one. Normalized
    per-cadence uncertainties, including missing errors, remain unchanged by
    detrending. All operations use native array columns and Astropy utilities.
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    from astropy.time import Time
    from astropy import units
    from scipy.signal import medfilt
    from tdpy.verbosity import print

    light_curves = [as_light_curve(curve) for curve in light_curves]
    if not light_curves:
        raise ValueError('At least one light curve is required')
    if plots:
        light_curves[0].plot()
        plot_path = Path('presentation_plots') / f'{target_label}_rawlc.png'
        print(f'Writing to {plot_path}...')
        plt.savefig(plot_path)
        plt.close()
    normalized_curves = []
    max_day = 0  # [day]
    for curve in light_curves:
        curve = curve.remove_nans()
        if plots and not normalized_curves:
            max_day = curve.time[-1].value
            curve.plot()
            plot_path = Path('presentation_plots') / f'{target_label}_flattenedlc.png'
            print(f'Writing to {plot_path}...')
            plt.savefig(plot_path)
            plt.close()
        curve = curve.normalize()
        normalized_curves.append(curve.remove_outliers())
        if report:
            print(f'Appended LC with {len(curve.time)} points.')
    normalized_curves = [curve.normalize() for curve in normalized_curves]
    first = normalized_curves[0]
    if isinstance(first.time, Time):
        if not all(isinstance(curve.time, Time) for curve in normalized_curves):
            raise ValueError('Cannot stitch absolute and relative time coordinates')
        time = Time(np.concatenate([curve.time.to_value('jd') for curve in normalized_curves]),
                    format='jd', scale=first.time.scale)
        time.format = first.time.format
    else:
        time = np.concatenate([curve.time.to_value(units.day) for curve in normalized_curves]) * units.day
    curve = LightCurve(time, np.concatenate([curve.flux for curve in normalized_curves]),
                       np.concatenate([curve.flux_err for curve in normalized_curves]))
    trend = medfilt(curve.flux, kernel_size=501)
    detrended_flux = curve.flux - trend if injection else curve.flux - trend + 1
    if plots:
        print('Plotting detrended LC...')
        curve_plot = curve[(curve.time.value > 0) & (curve.time.value < max_day)]
        plt.figure(figsize=(10, 5))
        curve_plot.scatter(s=1, c='k', label='Raw LC')
        plt.xlabel('Time [days]')
        plt.ylabel('Flux')
        plt.title(f'ID {target_label} Detrending')
        plt.legend()
        plot_path = Path('presentation_plots') / f'{target_label}_detrendedlc.png'
        print(f'Writing to {plot_path}...')
        plt.savefig(plot_path)
        plt.close()
        print('Plotted detrended LC.')
    return LightCurve(time=curve.time, flux=detrended_flux, flux_err=curve.flux_err)


preprocess_tess_lightkurve = preprocess_tess_light_curve
prepare_lightkurve_collection = prepare_light_curve_collection


def light_curve_from_image_stack(stack, aperture, *, normalize=True):
    """Return time [day], aperture flux, and one-sigma error from Lygos.

    Times retain the stack's original epoch and time system. Normalization divides
    flux and error by the same positive median, without fitting a trend or changing
    uncertainties to match the scatter. Pixel errors must be supplied and are
    combined as independent uncertainties by Lygos. Subtract any background
    before calling this adapter. It does not add uncertainty from a background
    estimate, so the caller must account for that uncertainty in the supplied
    errors. Frame selection also belongs in the input stack.
    No mission response or noise prescription is inferred from the mission label.
    """
    if stack.error is None:
        raise ValueError('Image-stack photometry requires supplied pixel uncertainties')
    if not isinstance(normalize, (bool, np.bool_)):
        raise ValueError('normalize must be Boolean')
    inputs = (stack.flux, stack.error, stack.time, stack.quality, aperture)
    if any(has_masked_or_complex_values(values) for values in inputs):
        raise ValueError('Image-stack photometry requires unmasked real inputs')
    flux = np.asarray(stack.flux, dtype=float)
    error = np.asarray(stack.error, dtype=float)
    time = np.asarray(stack.time, dtype=float)
    quality = np.asarray(stack.quality)
    aperture = np.asarray(aperture)
    if (flux.ndim != 3 or error.shape != flux.shape or time.shape != (len(flux),)
            or quality.shape != time.shape or len(time) < 3):
        raise ValueError('Image-stack photometry requires aligned cubes and at least three frames')
    if aperture.dtype != np.dtype(bool) or aperture.shape != flux.shape[1:] or not aperture.any():
        raise ValueError('aperture must be a nonempty Boolean mask matching the image shape')
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError('Image-stack times must be finite and strictly increasing')
    if np.any(quality != 0):
        raise ValueError('Select good-quality frames explicitly before extracting photometry')
    if (not np.isfinite(flux[:, aperture]).all() or not np.isfinite(error[:, aperture]).all()
            or np.any(error[:, aperture] < 0)):
        raise ValueError('Aperture pixels require finite flux and nonnegative uncertainties')
    photometry = stack.aperture_photometry(aperture)
    flux = np.asarray(photometry['flux'], dtype=float)
    error = np.asarray(photometry['flux_error'], dtype=float)
    if (flux.shape != time.shape or error.shape != time.shape or not np.isfinite(flux).all()
            or not np.isfinite(error).all() or np.any(error <= 0)):
        raise ValueError('Extracted photometry requires finite flux and positive uncertainties')
    if normalize:
        median = np.median(flux)
        if median <= 0:
            raise ValueError('Relative photometry requires a positive median aperture flux')
        flux = flux / median
        error = error / median
    return np.column_stack((time, flux, error))