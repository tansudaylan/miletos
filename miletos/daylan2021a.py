"""Reproduction of the TESS transit analysis of HD 108236 (TOI-1233) by Daylan et al. (2021a).

PCAT samples a four-planet transit model of the TESS Sectors 10 and 11 SPOC light curves that
Miletos has retrieved and detrended. Each transit window has its own linear baseline, which the
likelihood marginalizes together with white-noise jitter. Planet radii follow from the radius
ratios and the stellar radius of the paper.
"""

from tdpy.verbosity import print

import numpy as np

from .paths import get_data_path, get_repository_path

NAMEPLAN = ('b', 'c', 'd', 'e')
# published ephemerides and radius ratios, used to center the priors
EPOC_PUBL = np.array([2458572.1128, 2458572.3949, 2458571.3368, 2458586.5677])  # [BJD]
PERI_PUBL = np.array([3.79523, 6.20370, 14.17555, 19.5917])  # [day]
RRAT_PUBL = np.array([0.01638, 0.02134, 0.02805, 0.0323])
# Daylan et al. (2021a) results: median, lower, upper
RADI_PUBL = np.array([[1.586, 0.098, 0.098], [2.068, 0.091, 0.10], [2.72, 0.11, 0.11], [3.12, 0.12, 0.13]])  # [R_earth]
PERI_PUBL_UNCE = np.array([[0.00044, 0.00047], [0.00052, 0.00064], [0.0011, 0.00099], [0.0020, 0.0022]])  # [day]
RADI_STAR = 0.888  # [R_sun]
STDV_RADI_STAR = 0.017  # [R_sun]
RADI_SUN_EART = 109.076  # solar radius in Earth radii
DURA_WIND = 0.35  # [day], half width of the window fitted around each transit
LABL_PARA = ('$T_0$', '$P$', '$R_p/R_\\star$', '$(R_\\star+R_p)/a$', '$\\cos i$')


def retr_names():
    names = [f'{name}{plan}' for plan in NAMEPLAN for name in ('epoc', 'peri', 'rrat', 'rsma', 'cosi')]
    return names + ['qlmdk1', 'qlmdk2']


def retr_lcur():
    """Return times [BJD], relative fluxes, uncertainties, and window indices of data near transits."""
    listarry = []
    for sector in (10, 11):
        path = get_data_path() / 'TOI-1233' / 'data' / f'RelativeFlux_DataCube_Detrended_TESS_Sector{sector}_2min.csv'
        print(f'Reading from {path}...')
        listarry.append(np.loadtxt(path, delimiter=','))
    time, flux, stdv = np.concatenate(listarry).T
    good = np.isfinite(time) & np.isfinite(flux) & np.isfinite(stdv)
    time, flux, stdv = time[good], flux[good], stdv[good]
    # each window is one transit of one planet; overlapping windows are merged
    epocwind = np.sort(np.concatenate([epoc + peri * np.arange(np.ceil((time.min() - epoc) / peri),
                                                                np.floor((time.max() - epoc) / peri) + 1)
                                       for epoc, peri in zip(EPOC_PUBL, PERI_PUBL)]))
    indxwind = np.full(time.size, -1)
    for k, epoc in enumerate(epocwind):
        indx = np.abs(time - epoc) < DURA_WIND
        indxwind[indx & (indxwind < 0)] = k
    keep = indxwind >= 0
    time, flux, stdv, indxwind = time[keep], flux[keep], stdv[keep], indxwind[keep]
    indxwind = np.unique(indxwind, return_inverse=True)[1]
    # drop windows with too few points to constrain a baseline
    numb = np.bincount(indxwind)
    keep = numb[indxwind] > 30
    return time[keep], flux[keep], stdv[keep], np.unique(indxwind[keep], return_inverse=True)[1]


def retr_matrblin(time, indxwind):
    """Linear baseline basis with two columns per transit window."""
    columns = []
    for k in np.unique(indxwind):
        indx = indxwind == k
        timescal = np.where(indx, time - time[indx].mean(), 0.)
        columns += [indx.astype(float), timescal]
    return np.column_stack(columns)


def retr_modl(values, time, timeexpo=None):
    """Relative flux of four transiting planets with a shared quadratic limb darkening."""
    import batman
    from tdpy.exoplanet import kipping_to_quadratic_limb_darkening

    coeflmdk = list(kipping_to_quadratic_limb_darkening(values[-2], values[-1]))
    rflx = np.ones_like(time)
    supersample = {} if timeexpo is None else dict(supersample_factor=3, exp_time=timeexpo)
    for epoc, peri, rrat, rsma, cosi in values[:-2].reshape(len(NAMEPLAN), 5):
        para = batman.TransitParams()
        para.t0, para.per, para.rp, para.a = epoc, peri, rrat, (1. + rrat) / rsma
        para.inc, para.ecc, para.w = np.degrees(np.arccos(cosi)), 0., 90.
        para.limb_dark, para.u = 'quadratic', coeflmdk
        rflx += batman.TransitModel(para, time, **supersample).light_curve(para) - 1.
    return rflx


def retr_llik(values, state):
    """Log-likelihood marginalized over per-window linear baselines and white-noise jitter."""
    from pcat.radial_velocity import retr_llik_rvelmarg

    modl = retr_modl(values, state['time'], state['timeexpo'])
    return retr_llik_rvelmarg(state['flux'], state['stdv'], modl[:, None] * state['matrblin'], state['listjitt'])


def fit_daylan2021a_transits(numbchan=4, numbsamp=100000, numbburn=100000, seed=0, boolreus=False):
    """Sample the four-planet posterior with PCAT; return samples, radii [R_earth], and the data state.

    With boolreus, a previously saved posterior is returned instead of sampling again.
    """
    from pcat.fixed import sample_fixed_chains

    time, flux, stdv, indxwind = retr_lcur()
    state = dict(time=time, flux=flux, stdv=stdv, matrblin=retr_matrblin(time, indxwind),
                 listjitt=np.geomspace(1e-5, 3e-4, 4), timeexpo=2. / 1440.)
    path = get_repository_path() / 'examples' / 'TOI-1233' / 'data' / 'daylan2021a_transit_posterior.npz'
    if boolreus and path.exists():
        print(f'Reading from {path}...')
        arry = np.load(path)
        return arry['samples'], arry['radi'], state
    names = retr_names()
    minima, maxima, initial = [], [], []
    for epoc, peri, rrat in zip(EPOC_PUBL, PERI_PUBL, RRAT_PUBL):
        minima += [epoc - 0.05, peri - 0.01, 0.3 * rrat, 0.01, 0.]
        maxima += [epoc + 0.05, peri + 0.01, 2. * rrat, 0.2, 0.2]
        initial += [epoc, peri, rrat, 0.05, 0.02]
    minima, maxima, initial = np.array(minima + [0., 0.]), np.array(maxima + [1., 1.]), np.array(initial + [0.3, 0.3])
    chain, _ = sample_fixed_chains(state, retr_llik, None, names, ['self'] * len(names), minima, maxima,
                                   np.zeros(len(names)), np.ones(len(names)), initial[None, :], numbchan, numbsamp,
                                   numbburn, booladaptstdp=True, booldiag=False, seed=seed)
    samples = chain.reshape(-1, len(names))
    rng = np.random.default_rng(seed)
    radistar = rng.normal(RADI_STAR, STDV_RADI_STAR, samples.shape[0])  # [R_sun]
    radi = samples[:, 2:-2:5] * radistar[:, None] * RADI_SUN_EART  # [R_earth]
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Writing to {path}...')
    np.savez(path, samples=samples, chain=chain, radi=radi)
    return samples, radi, state


def retr_summary_daylan2021a(samples, radi):
    """Return text lines of the posterior radii and periods next to the published values."""
    lines = []
    for k, plan in enumerate(NAMEPLAN):
        lowr, medi, uppr = np.percentile(radi[:, k], [16., 50., 84.])
        peri = np.percentile(samples[:, 5 * k + 1], [16., 50., 84.])
        publ = RADI_PUBL[k]
        lines.append(f'{plan}: radius {medi:.3f} (+{uppr - medi:.2f} / -{medi - lowr:.2f}) R_earth, published '
                     f'{publ[0]} (+{publ[2]} / -{publ[1]}); period {peri[1]:.5f} (+{peri[2] - peri[1]:.5f} / '
                     f'-{peri[1] - peri[0]:.5f}) day, published {PERI_PUBL[k]}')
    return lines


def plot_daylan2021a_transits(samples, radi, state, typefileplot='png'):
    """Write the reproduction figures of Daylan et al. (2021a) under examples/TOI-1233/visuals."""
    import matplotlib.pyplot as plt
    from pcat.plotting import plot_grid
    from .visualization import miletos_plot_context

    path_visu = get_repository_path() / 'examples' / 'TOI-1233' / 'visuals'
    path_visu.mkdir(parents=True, exist_ok=True)
    time, flux, stdv = state['time'], state['flux'], state['stdv']
    medi = np.median(samples, 0)
    rng = np.random.default_rng(1)
    draws = samples[rng.choice(samples.shape[0], 200, replace=False)]
    colors = ('#A51C30', '#1f77b4', '#2ca02c', '#ff7f0e')

    def save(figure, name):
        path = path_visu / f'daylan2021a_{name}.{typefileplot}'
        print(f'Writing to {path}...')
        figure.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(figure)

    # detrend each window with its best-fitting baseline for display
    modlmedi = retr_modl(medi, time, state['timeexpo'])
    matrdesi = modlmedi[:, None] * state['matrblin']
    coef = np.linalg.lstsq(matrdesi / stdv[:, None], flux / stdv, rcond=None)[0]
    detr = flux / (state['matrblin'] @ coef)

    with miletos_plot_context():
        # phase-folded transits of each planet with the other planets removed
        figure, axes = plt.subplots(1, len(NAMEPLAN), figsize=(7.1, 2.6), sharey=True)
        for p, (axis, plan) in enumerate(zip(axes, NAMEPLAN)):
            epoc, peri = medi[5 * p], medi[5 * p + 1]
            othr = medi.copy()
            othr[5 * p + 2] = 1e-6
            resi = detr - retr_modl(othr, time, state['timeexpo']) + 1.
            offs = 24. * (np.mod(time - epoc + 0.5 * peri, peri) - 0.5 * peri)  # [hour]
            indx = np.abs(offs) < 6.
            axis.plot(offs[indx], 1e6 * (resi[indx] - 1.), '.', ms=1, color='0.7')
            binsoffs = np.linspace(-6., 6., 49)
            indxbins = np.digitize(offs[indx], binsoffs) - 1
            meanbins = [np.mean(resi[indx][indxbins == k]) for k in range(48)]
            axis.plot(0.5 * (binsoffs[1:] + binsoffs[:-1]), 1e6 * (np.array(meanbins) - 1.), 'o', ms=2.5, color='black')
            offsfine = np.linspace(-6., 6., 400)
            modl = []
            for draw in draws:
                solo = np.concatenate([np.array([draw[5 * p], draw[5 * p + 1], draw[5 * p + 2], draw[5 * p + 3],
                                                 draw[5 * p + 4]]) if q == p else
                                       np.array([draw[5 * q], draw[5 * q + 1], 1e-6, draw[5 * q + 3], draw[5 * q + 4]])
                                       for q in range(len(NAMEPLAN))] + [draw[-2:]])
                modl.append(retr_modl(solo, draw[5 * p] + offsfine / 24.) - 1.)
            axis.fill_between(offsfine, *1e6 * np.percentile(modl, [16., 84.], 0), color=colors[p], alpha=0.5, lw=0)
            axis.plot(offsfine, 1e6 * np.median(modl, 0), color=colors[p], lw=1.)
            axis.set_title(f'HD 108236 {plan}, P = {peri:.3f} day')
            axis.set_xlabel('Time from mid-transit [hour]')
        axes[0].set_ylim(-1600., 800.)
        axes[0].set_ylabel('Relative flux [ppm]')
        figure.subplots_adjust(wspace=0.05)
        save(figure, 'phase_folded_transits')

        # radii and periods against the published values
        figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
        posi = np.arange(len(NAMEPLAN))
        quan = np.percentile(radi, [16., 50., 84.], 0)
        axes[0].errorbar(posi - 0.1, quan[1], [quan[1] - quan[0], quan[2] - quan[1]], fmt='o', color='#A51C30',
                         label='This reproduction')
        axes[0].errorbar(posi + 0.1, RADI_PUBL[:, 0], RADI_PUBL[:, 1:].T, fmt='s', color='black',
                         label='Daylan et al. (2021a)')
        axes[0].set_xticks(posi, NAMEPLAN)
        axes[0].set_xlabel('Planet')
        axes[0].set_ylabel('Radius [$R_\\oplus$]')
        axes[0].legend(loc='upper left')
        quanperi = np.percentile(samples[:, 1:-2:5], [16., 50., 84.], 0)
        stdvpubl = PERI_PUBL_UNCE.mean(1)
        axes[1].errorbar(posi, (quanperi[1] - PERI_PUBL) / stdvpubl, [(quanperi[1] - quanperi[0]) / stdvpubl,
                         (quanperi[2] - quanperi[1]) / stdvpubl], fmt='o', color='#A51C30')
        axes[1].axhspan(-1., 1., color='0.8', label='Published $\\pm1\\sigma$')
        axes[1].axhline(0., color='black', lw=0.8)
        axes[1].set_xticks(posi, NAMEPLAN)
        axes[1].set_xlabel('Planet')
        axes[1].set_ylabel('Period offset [published $\\sigma$]')
        axes[1].legend(loc='upper left')
        figure.tight_layout()
        save(figure, 'published_comparison')

    plot_grid(str(path_visu) + '/', 'daylan2021a_radius_posterior', radi,
              [f'$R_{plan}$ [$R_\\oplus$]' for plan in NAMEPLAN], typefileplot=typefileplot)
