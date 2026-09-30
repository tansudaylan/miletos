"""Miletos workflow for the WASP-121 b TESS phase-curve paper."""

from tdpy.verbosity import print
import os
from pathlib import Path

import numpy as np

from . import main
from .paths import get_repository_path


def _load_cached_sector7(example_path: Path):
    """Return the cached TESS Sector 7 PDCSAP light curve as a Miletos input cube, or None."""
    path = example_path / 'data' / 'TESS_PDCSAP_FLUX_Sector07.csv'
    if not path.exists():
        return None
    print(f'Reading from {path}...')
    photometry = np.loadtxt(path, delimiter=',')
    photometry = photometry[np.isfinite(photometry).all(axis=1)]
    return {'Raw': [[[photometry[:, None, :]]], []]}


def run_daylan2021b_reproduction(typefileplot: str = 'png', fit: bool = True) -> dict:
    """Analyze the public TESS Sector 7 light curve with Miletos.

    A cached Sector 7 light curve is analyzed offline; otherwise Miletos retrieves it from MAST.
    """
    if typefileplot not in {'png', 'pdf'}:
        raise ValueError("typefileplot must be 'png' or 'pdf'")

    example_path = get_repository_path() / 'examples' / 'WASP-121b'
    cached = _load_cached_sector7(example_path)
    source = dict(liststrgtypedata=[['obsd'], []], listtsecsele=[7], boolforcoffl=False)
    if cached is not None:
        source = dict(liststrgtypedata=[['inpt'], []], listarrytser=cached, boolforcoffl=True)
    return main.init(
        strgmast='WASP-121',
        strgtarg='WASP-121b-Daylan2021b',
        labltarg='WASP-121 b',
        boolfitt=fit,
        boolplot=True,
        boolplottser=True,
        typefileplot=typefileplot,
        typeplotback='white',
        typeverb=0,
        listlablinst=[['TESS'], []],
        boolbdtr=[[False], []],
        typepriocomp='exar',
        dictfitt={'typemodl': 'PlanetarySystemEmittingCompanion'},
        pathtarg=os.path.join(str(example_path), ''),
        **source,
    )


# ephemeris of Delrez et al. (2016), used only to center the priors
EPOC_LITE = 2456635.70832  # [BJD_TDB]
PERI_LITE = 1.2749255  # [day]
STDV_PERI_LITE = 2.5e-6  # [day]
TMPT_STAR = 6460.  # [K], Delrez et al. (2016)
STDV_TMPT_STAR = 140.  # [K]
# dayside and nightside TESS brightness temperatures of Daylan et al. (2021b): median, lower, upper [K]
PUBLISHED_TEMPERATURES = {'day': (3012., 42., 40.), 'night': (2022., 602., 254.)}
NAMES = ('epoc', 'peri', 'rratcomp', 'rsmacomp', 'cosicomp', 'qlmdk1', 'qlmdk2', 'fluxday', 'fluxnight', 'phasoffs',
         'amplelli')
LABELS = ('$T_0$ [BJD]', '$P$ [day]', '$R_p/R_\\star$', '$(R_\\star+R_p)/a$', '$\\cos i$', '$q_1$', '$q_2$',
          'Dayside flux [ppm]', 'Nightside flux [ppm]', 'Phase offset [deg]', 'Ellipsoidal [ppm]')


def retr_lcur_sector7(example_path, cadebins=10.):
    """Return binned Sector 7 times [BJD], relative fluxes, uncertainties, and TESS orbit indices.

    cadebins is the bin width [minute].
    """
    path = example_path / 'data' / 'TESS_PDCSAP_FLUX_Sector07.csv'
    print(f'Reading from {path}...')
    time, flux, stdv = np.loadtxt(path, delimiter=',', unpack=True)
    boolgood = np.isfinite(time) & np.isfinite(flux) & np.isfinite(stdv)
    time, flux, stdv = time[boolgood], flux[boolgood], stdv[boolgood]
    indxbins = np.floor((time - time[0]) / (cadebins / 1440.)).astype(int)
    _, indxinve, numb = np.unique(indxbins, return_inverse=True, return_counts=True)
    # keep bins with most of their 2-minute exposures
    boolfull = numb[indxinve] >= 0.8 * cadebins / 2.
    time, flux, stdv, indxinve = time[boolfull], flux[boolfull], stdv[boolfull], indxinve[boolfull]
    _, indxinve, numb = np.unique(indxinve, return_inverse=True, return_counts=True)
    timebins = np.bincount(indxinve, time) / numb
    fluxbins = np.bincount(indxinve, flux) / numb
    stdvbins = np.sqrt(np.bincount(indxinve, stdv**2)) / numb
    indxorbt = np.concatenate(([0], np.cumsum(np.diff(timebins) > 1.)))
    return timebins, fluxbins, stdvbins, indxorbt


def retr_matrblin(time, indxorbt):
    """Quadratic baseline basis per TESS orbit, one set of columns per orbit."""
    columns = []
    for k in np.unique(indxorbt):
        indx = indxorbt == k
        timescal = np.zeros_like(time)
        timescal[indx] = (time[indx] - time[indx].mean()) / np.ptp(time[indx])
        for ordr in range(3):
            columns.append(np.where(indx, timescal**ordr, 0.))
    return np.column_stack(columns)


def retr_modl(values, time, timeexpo=None):
    """Planet-only relative flux model for a parameter vector (flux quantities in ppm, offset in deg)."""
    from tdpy.exoplanet import kipping_to_quadratic_limb_darkening
    from .phase_curve import retr_rflx_pcur

    epoc, peri, rratcomp, rsmacomp, cosicomp, qlmdk1, qlmdk2, fluxday, fluxnight, phasoffs, amplelli = values
    coeflmdk = np.array(kipping_to_quadratic_limb_darkening(qlmdk1, qlmdk2))
    return retr_rflx_pcur(time, epoc, peri, rratcomp, rsmacomp, cosicomp, coeflmdk, 1e-6 * fluxday, 1e-6 * fluxnight,
                          np.deg2rad(phasoffs), 1e-6 * amplelli, timeexpo=timeexpo)


def retr_llik(values, state):
    """Log-likelihood marginalized over the per-orbit multiplicative baselines and white-noise jitter."""
    from pcat.radial_velocity import retr_llik_rvelmarg

    modl = retr_modl(values, state['time'], state['timeexpo'])
    return retr_llik_rvelmarg(state['flux'], state['stdv'], modl[:, None] * state['matrblin'], state['listjitt'])


def retr_blin(values, state):
    """Best-fitting multiplicative baseline for display."""
    modl = retr_modl(values, state['time'], state['timeexpo'])
    matrdesi = modl[:, None] * state['matrblin']
    coef = np.linalg.lstsq(matrdesi / state['stdv'][:, None], state['flux'] / state['stdv'], rcond=None)[0]
    return state['matrblin'] @ coef


def fit_daylan2021b_phase_curve(numbchan=4, numbsamp=2000, numbburn=10000, seed=0, boolreus=False):
    """Sample the Sector 7 phase-curve posterior with PCAT and return samples, derived temperatures, and state.

    With boolreus, a previously saved posterior is returned instead of sampling again.
    """
    from pcat.fixed import sample_fixed_chains
    from .phase_curve import retr_brightness_temperature, retr_tess_response

    example_path = get_repository_path() / 'examples' / 'WASP-121b'
    time, flux, stdv, indxorbt = retr_lcur_sector7(example_path)
    state = dict(time=time, flux=flux, stdv=stdv, matrblin=retr_matrblin(time, indxorbt),
                 listjitt=np.geomspace(1e-6, 1e-3, 12), timeexpo=10. / 1440.)
    path = example_path / 'data' / 'daylan2021b_phase_curve_posterior.npz'
    if boolreus and path.exists():
        print(f'Reading from {path}...')
        arry = np.load(path)
        return arry['samples'], {'day': arry['tmptday'], 'night': arry['tmptnight']}, state
    epoc = EPOC_LITE + np.round((np.median(time) - EPOC_LITE) / PERI_LITE) * PERI_LITE  # [BJD]
    minima = np.array([epoc - 0.02, 0., 0.10, 0.20, 0., 0., 0., 0., -300., -60., -200.])
    maxima = np.array([epoc + 0.02, 2., 0.14, 0.40, 0.3, 1., 1., 1500., 1000., 60., 300.])
    means = np.zeros(len(NAMES))
    stdvs = np.ones(len(NAMES))
    means[1], stdvs[1] = PERI_LITE, STDV_PERI_LITE
    scales = ['self'] * len(NAMES)
    scales[1] = 'gaus'
    initial = np.array([[epoc, PERI_LITE, 0.1245, 0.29, 0.05, 0.3, 0.3, 400., 100., 0., 50.]])
    chain, _ = sample_fixed_chains(state, retr_llik, None, NAMES, scales, minima, maxima, means, stdvs, initial,
                                   numbchan, numbsamp, numbburn, booladaptstdp=True, booldiag=False, seed=seed)
    samples = chain.reshape(-1, len(NAMES))

    # propagate the stellar temperature uncertainty into the planetary brightness temperatures
    wavelength, response = retr_tess_response()
    rng = np.random.default_rng(seed)
    indxdraw = rng.choice(samples.shape[0], min(2000, samples.shape[0]), replace=False)
    tmptstar = rng.normal(TMPT_STAR, STDV_TMPT_STAR, indxdraw.size)
    tmpt = {name: retr_brightness_temperature(1e-6 * samples[indxdraw, NAMES.index('flux' + name)],
                                              samples[indxdraw, NAMES.index('rratcomp')], tmptstar, wavelength, response)
            for name in ('day', 'night')}
    print(f'Writing to {path}...')
    np.savez(path, samples=samples, chain=chain, tmptday=tmpt['day'], tmptnight=tmpt['night'])
    return samples, tmpt, state


def retr_summary_daylan2021b(samples, tmpt):
    """Return text lines of the posterior phase-curve parameters and temperatures next to the published values."""
    lines = []
    for name in ('fluxday', 'fluxnight', 'phasoffs', 'amplelli', 'rratcomp'):
        lowr, medi, uppr = np.percentile(samples[:, NAMES.index(name)], [16., 50., 84.])
        lines.append(f'{LABELS[NAMES.index(name)]}: {medi:.4g} (+{uppr - medi:.2g} / -{medi - lowr:.2g})')
    for name, (medipubl, lowrpubl, upprpubl) in PUBLISHED_TEMPERATURES.items():
        lowr, medi, uppr = np.percentile(tmpt[name][tmpt[name] > 0.], [16., 50., 84.])
        lines.append(f'{name}side brightness temperature: {medi:.0f} (+{uppr - medi:.0f} / -{medi - lowr:.0f}) K, '
                     f'published {medipubl:.0f} (+{upprpubl:.0f} / -{lowrpubl:.0f}) K')
    return lines


def plot_daylan2021b_phase_curve(samples, tmpt, state, typefileplot='png'):
    """Write the reproduction figures of Daylan et al. (2021b) under examples/WASP-121b/visuals."""
    import matplotlib.pyplot as plt
    from pcat.plotting import plot_grid
    from .visualization import miletos_plot_context

    path_visu = get_repository_path() / 'examples' / 'WASP-121b' / 'visuals'
    path_visu.mkdir(parents=True, exist_ok=True)
    time, flux, stdv = state['time'], state['flux'], state['stdv']
    medi = np.median(samples, 0)
    blin = retr_blin(medi, state)
    rng = np.random.default_rng(1)
    draws = samples[rng.choice(samples.shape[0], 200, replace=False)]

    def save(figure, name):
        path = path_visu / f'daylan2021b_{name}.{typefileplot}'
        print(f'Writing to {path}...')
        figure.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(figure)

    with miletos_plot_context():
        # raw light curve and the fitted stellar baseline
        figure, axis = plt.subplots(figsize=(7.1, 2.8))
        axis.plot(time - 2458490., flux, '.', ms=1.5, color='0.4', label='TESS Sector 7, 10 min bins (SPOC PDCSAP)')
        axis.plot(time - 2458490., blin * retr_modl(medi, time), color='#A51C30', lw=0.8, label='Median model')
        axis.set_xlabel('Time [BJD - 2458490]')
        axis.set_ylabel('Relative flux')
        axis.legend(loc='lower left')
        save(figure, 'light_curve')

        # phase-folded, detrended light curve showing transit, phase variation, and secondary eclipse
        phas = np.mod((time - medi[0]) / medi[1] + 0.25, 1.) - 0.25
        detr = (flux / blin - 1.) * 1e6  # [ppm]
        phasfine = np.linspace(-0.25, 0.75, 2000)
        modl = np.array([(retr_modl(draw, medi[0] + phasfine * medi[1]) - 1.) * 1e6 for draw in draws])
        binsphas = np.linspace(-0.25, 0.75, 81)
        indxbins = np.digitize(phas, binsphas) - 1
        meanbins = np.array([detr[indxbins == k].mean() for k in range(80)])
        stdvbins = np.array([detr[indxbins == k].std() / np.sqrt(max((indxbins == k).sum(), 1)) for k in range(80)])
        figure, axes = plt.subplots(2, 1, figsize=(7.1, 5.), sharex=True, gridspec_kw={'height_ratios': [1, 1.3]})
        for axis, ylim in zip(axes, [None, (-250., 650.)]):
            axis.plot(phas, detr, '.', ms=1, color='0.75', label='10 min bins')
            axis.errorbar(0.5 * (binsphas[1:] + binsphas[:-1]), meanbins, stdvbins, fmt='o', ms=3, color='black',
                          label='Phase bins')
            axis.fill_between(phasfine, *np.percentile(modl, [16., 84.], 0), color='#A51C30', alpha=0.4, lw=0,
                              label='PCAT 68% interval')
            axis.plot(phasfine, np.median(modl, 0), color='#A51C30', lw=1., label='PCAT median model')
            axis.set_ylabel('Relative flux [ppm]')
            if ylim is not None:
                axis.set_ylim(ylim)
        axes[0].legend(loc='lower right')
        axes[1].set_xlabel('Orbital phase')
        figure.subplots_adjust(hspace=0.05)
        save(figure, 'phase_curve')

    # joint posterior of the phase-curve parameters
    indxpara = [NAMES.index(name) for name in ('fluxday', 'fluxnight', 'phasoffs', 'amplelli', 'rratcomp')]
    plot_grid(str(path_visu) + '/', 'daylan2021b_phase_curve_posterior', samples[:, indxpara],
              [LABELS[k] for k in indxpara], typefileplot=typefileplot)

    with miletos_plot_context():
        # brightness temperatures against the published values
        figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
        for axis, name, labl in zip(axes, ('day', 'night'), ('Dayside', 'Nightside')):
            valu = tmpt[name][tmpt[name] > 0.]
            axis.hist(valu, bins=40, density=True, histtype='stepfilled', color='#A51C30', alpha=0.6,
                      label='This reproduction')
            medipubl, lowrpubl, upprpubl = PUBLISHED_TEMPERATURES[name]
            axis.axvspan(medipubl - lowrpubl, medipubl + upprpubl, color='0.5', alpha=0.4,
                         label='Daylan et al. (2021b)')
            axis.axvline(medipubl, color='black', lw=1.)
            axis.set_xlabel(f'{labl} TESS brightness temperature [K]')
            axis.set_yticks([])
            if name == 'night':
                axis.text(0.03, 0.95, '%.0f%% of samples have\nnonpositive flux' %
                          (100. * np.mean(tmpt[name] <= 0.)), transform=axis.transAxes, va='top')
        axes[0].legend(loc='upper left')
        save(figure, 'brightness_temperatures')
