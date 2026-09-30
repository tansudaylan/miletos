"""Phase-curve modeling of emitting planets in broadband photometry.

The model multiplies a slowly varying stellar baseline with the sum of a limb-darkened transit,
the planet's phase-dependent thermal and reflected emission hidden during secondary eclipse, and
the ellipsoidal variation of the tidally distorted star. Brightness temperatures follow from the
planet-to-star flux ratio integrated over the instrument response.
"""

from tdpy.verbosity import print

from urllib.request import urlopen

import numpy as np
from scipy.optimize import brentq

from .paths import get_data_path

TESS_RESPONSE_URL = 'https://heasarc.gsfc.nasa.gov/docs/tess/data/tess-response-function-v2.0.csv'
PLANCK = 6.62607015e-34  # [J s]
LIGHT_SPEED = 2.99792458e8  # [m/s]
BOLTZMANN = 1.380649e-23  # [J/K]


def retr_tess_response():
    """Return TESS wavelengths [m] and relative response, downloading and caching the public curve once."""
    path = get_data_path() / 'tess-response-function-v2.0.csv'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Writing to {path}...')
        with urlopen(TESS_RESPONSE_URL, timeout=60) as response:
            path.write_bytes(response.read())
    print(f'Reading from {path}...')
    table = np.loadtxt(path, delimiter=',', comments='#')
    return 1e-9 * table[:, 0], table[:, 1]


def retr_band_planck(temperature, wavelength, response):
    """Photon-weighted Planck intensity integrated over a response curve [arbitrary units]."""
    temperature = np.atleast_1d(np.asarray(temperature, dtype=float))
    exponent = PLANCK * LIGHT_SPEED / (wavelength[None, :] * BOLTZMANN * temperature[:, None])
    # photon-counting detectors weight the energy intensity by wavelength
    intensity = wavelength[None, :]**-4 / np.expm1(exponent)
    return np.trapezoid(intensity * response[None, :], wavelength, axis=1)


def retr_brightness_temperature(ratiflux, rratcomp, tmptstar, wavelength, response):
    """Band brightness temperature [K] of a planet from its flux ratio and radius ratio.

    Nonpositive flux ratios have no brightness temperature and return zero.
    """
    ratiflux, rratcomp, tmptstar = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (ratiflux, rratcomp, tmptstar)))
    tmptcomp = np.zeros(ratiflux.shape)
    for k in np.ndindex(ratiflux.shape):
        if ratiflux[k] <= 0.:
            continue
        target = ratiflux[k] / rratcomp[k]**2 * retr_band_planck(tmptstar[k], wavelength, response)[0]
        tmptcomp[k] = brentq(lambda tmpt: retr_band_planck(tmpt, wavelength, response)[0] - target, 50., 2e4)
    return tmptcomp


def retr_rflx_pcur(time, epoc, peri, rratcomp, rsmacomp, cosicomp, coeflmdk, fluxday, fluxnight, phasoffs, amplelli,
                   timeexpo=None):
    """Relative flux of a transiting, emitting planet on a circular orbit, before the stellar baseline.

    Flux ratios and amplitudes are relative to the stellar flux. The planet's brightness peaks at
    orbital phase 0.5 - phasoffs / (2 pi), so an eastward hotspot has a positive phasoffs [rad].
    Transits and eclipses are evaluated analytically with batman (Kreidberg 2015), supersampled
    over the exposure time timeexpo [day] when given.
    """
    import batman

    para = batman.TransitParams()
    para.t0, para.per, para.rp = epoc, peri, rratcomp
    para.a = (1. + rratcomp) / rsmacomp
    para.inc = np.degrees(np.arccos(cosicomp))
    para.ecc, para.w = 0., 90.
    para.limb_dark, para.u = 'quadratic', list(coeflmdk)
    para.fp, para.t_secondary = 1., epoc + 0.5 * peri
    supersample = {} if timeexpo is None else dict(supersample_factor=5, exp_time=timeexpo)
    rflx = batman.TransitModel(para, time, **supersample).light_curve(para)
    # with a unit planet-to-star flux ratio, the secondary-eclipse curve minus one is the visible planet fraction
    fracvisi = batman.TransitModel(para, time, transittype='secondary', **supersample).light_curve(para) - 1.
    phas = 2. * np.pi * (time - epoc) / peri  # [rad]
    fluxcomp = fluxnight + (fluxday - fluxnight) * 0.5 * (1. - np.cos(phas + phasoffs))
    return rflx + fluxcomp * fracvisi - amplelli * np.cos(2. * phas)
