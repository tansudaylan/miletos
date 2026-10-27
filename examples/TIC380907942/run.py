"""Inspect the observed TESS pulse discussed by Sorabella et al. (2026)."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tdpy.cli import add_plot_arguments
from tdpy.plotting import save_figure
from tdpy.verbosity import print
from miletos.paths import get_data_path, get_repository_path
from miletos.tess_transit_survey import download_qlp_light_curves, read_qlp_light_curve
from miletos.visualization import miletos_plot_context, plot_binned_light_curve


TIC_ID = 380907942
SECTOR = 30
PAPER_DOI = '10.3847/1538-4357/ae25f8'
DATA_DOI = '10.17909/mrrv-4980'
EVENT_REFERENCE_BTJD = 2118.0  # [BTJD day]
EVENT_HALF_WINDOW_DAYS = 3.0  # [day]


def six_cadence_summary(light_curve):
    """Return full six-point means and population standard deviations.

    The paper states six points per bin and standard-deviation errors. This
    summary explicitly uses ddof=0 and retains any incomplete tail in the raw
    output rather than treating it as a full six-point bin.
    """
    values = light_curve[['time_btjd', 'flux']].to_numpy(dtype=float)
    if len(values) < 6 or not np.isfinite(values).all():
        raise ValueError('Need at least six finite cadences')
    if np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError('Cadences must have strictly increasing times')
    blocks = values[:len(values) // 6 * 6].reshape(-1, 6, 2)
    return pd.DataFrame({
        'time_btjd': blocks[:, :, 0].mean(axis=1),
        'flux': blocks[:, :, 1].mean(axis=1),
        'flux_scatter': blocks[:, :, 1].std(axis=1, ddof=0),
        'cadences_per_bin': 6,
    })


def run_analysis(*, input_path=None, download=False, time_tag=None, typeplotback='white', typefileplot='png'):
    """Publish observed photometry, not a joint self-lensing/RV reproduction."""
    time_tag = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') if time_tag is None else time_tag
    if not isinstance(time_tag, str) or not re.fullmatch(r'[0-9]{8}_[0-9]{6}', time_tag):
        raise ValueError('time_tag must use YYYYMMDD_HHMMSS')
    datetime.strptime(time_tag, '%Y%m%d_%H%M%S')
    if typeplotback not in ('white', 'dark'):
        raise ValueError('typeplotback must be white or dark')
    if typefileplot not in ('png', 'pdf'):
        raise ValueError('typefileplot must be png or pdf')
    data_directory = get_data_path() / f'TIC{TIC_ID}'
    visual_directory = get_repository_path() / 'examples' / f'TIC{TIC_ID}' / 'visuals'
    paths = {
        'raw': data_directory / f'sector30_qlp_{time_tag}.csv',
        'binned': data_directory / f'sector30_six_cadences_{time_tag}.csv',
        'manifest': data_directory / f'sector30_provenance_{time_tag}.json',
        'figure': visual_directory / f'tic{TIC_ID}_sector30_observed_pulse_{time_tag}.{typefileplot}',
    }
    for path in paths.values():
        if path.exists():
            raise FileExistsError(path.name)
    if input_path is None:
        cache = data_directory / 'qlp_cache'
        if download:
            candidates = download_qlp_light_curves([TIC_ID], [SECTOR], cache)
        else:
            candidates = list(cache.rglob('hlsp_qlp_tess_ffi_s0030-0000000380907942_tess_v*_llc.fits'))
        if len(candidates) != 1:
            raise ValueError('Supply one exact Sector 30 QLP FITS product with --input or --download')
        input_path = candidates[0]
    input_path = Path(input_path)
    print(f'Reading from {input_path}...')
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    source_paths = (Path(__file__).resolve(), Path(read_qlp_light_curve.__code__.co_filename).resolve())
    source_hashes = {}
    for source_path in source_paths:
        print(f'Reading from {source_path}...')
        source_hashes[source_path.name] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    curve = read_qlp_light_curve(input_path, include_aperture_flux=True)
    if not curve['tic_id'].eq(TIC_ID).all() or not curve['sector'].eq(SECTOR).all():
        raise ValueError('The input must be TIC 380907942 Sector 30')
    aperture_curve = curve[['time_btjd', 'aperture_flux']].rename(columns={'aperture_flux': 'flux'})
    binned = six_cadence_summary(aperture_curve)
    near_event = binned['time_btjd'].between(
        EVENT_REFERENCE_BTJD-EVENT_HALF_WINDOW_DAYS,
        EVENT_REFERENCE_BTJD+EVENT_HALF_WINDOW_DAYS,
    )
    if near_event.sum() < 2:
        raise ValueError('The retained light curve does not cover the published pulse interval')
    with miletos_plot_context(typeplotback):
        figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), constrained_layout=True)
        foreground = 'black' if typeplotback == 'white' else 'white'
        axes[0].plot(curve['time_btjd'], curve['aperture_flux'], linestyle='none', marker='.',
                 markersize=2, color=foreground, label='QLP aperture flux')
        axes[0].plot(curve['time_btjd'], curve['flux'], linestyle='none', marker='.',
                 markersize=2, color='#64818A', alpha=0.6, label='QLP detrended flux')
        selected = binned.loc[near_event]
        plot_binned_light_curve(axes[1], selected['time_btjd'], selected['flux'], selected['flux_scatter'],
                                typeplotback=typeplotback, markersize=3, linewidth=0.6,
                                label='Six-cadence means and scatter')
        axes[0].set(title='TIC 380907942, TESS Sector 30', xlabel='Time [BTJD day]', ylabel='Normalized flux')
        axes[1].set(xlabel='Time [BTJD day]', ylabel='Normalized flux',
                    xlim=(EVENT_REFERENCE_BTJD-EVENT_HALF_WINDOW_DAYS, EVENT_REFERENCE_BTJD+EVENT_HALF_WINDOW_DAYS))
        figure.legend(loc='outside upper right', title='Sorabella et al. (2026), observed photometry')
        print(f'Reading from {input_path}...')
        if hashlib.sha256(input_path.read_bytes()).hexdigest() != input_hash:
            plt.close(figure)
            raise RuntimeError('Input changed during analysis; refusing to publish')
        for source_path in source_paths:
            print(f'Reading from {source_path}...')
            if hashlib.sha256(source_path.read_bytes()).hexdigest() != source_hashes[source_path.name]:
                plt.close(figure)
                raise RuntimeError('Source changed during analysis; refusing to publish')
        data_directory.mkdir(parents=True, exist_ok=True)
        print(f"Writing to {paths['raw']}...")
        curve.to_csv(paths['raw'], index=False)
        print(f"Writing to {paths['binned']}...")
        binned.to_csv(paths['binned'], index=False)
        save_figure(figure, paths['figure'], typefileplot=typefileplot,
                typeplotback=typeplotback, close_figure=True)
    manifest = {
        'time_tag_utc': time_tag, 'tic_id': TIC_ID, 'sector': SECTOR,
        'paper_doi': PAPER_DOI, 'mast_data_doi': DATA_DOI,
        'input_filename': input_path.name, 'input_sha256': input_hash,
        'source_sha256': source_hashes,
        'retained_cadences': len(curve), 'full_six_cadence_bins': len(binned),
        'unbinned_tail_cadences': len(curve) % 6,
        'bin_error_definition': 'Population standard deviation, ddof=0, not standard error of the mean',
        'displayed_flux': 'SAP_FLUX, independently median-normalized',
        'control_flux': 'Default reader flux, KSPSAP_FLUX for this product',
        'aperture_uncertainties': 'No per-cadence SAP_FLUX_ERR supplied; raw aperture points have no fabricated error bars',
        'figure': paths['figure'].name,
        'scope': 'Observed photometry only. No radial-velocity time series, joint-fit reproduction, physical lens fit or confirmation is claimed.',
    }
    print(f"Writing to {paths['manifest']}...")
    paths['manifest'].write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_arguments(parser)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--time-tag')
    parser.add_argument('--typeplotback', choices=('white', 'dark'), default='white')
    arguments = parser.parse_args()
    run_analysis(input_path=arguments.input, download=arguments.download,
                 time_tag=arguments.time_tag, typeplotback=arguments.typeplotback,
                 typefileplot=arguments.typefileplot)


if __name__ == '__main__':
    main()