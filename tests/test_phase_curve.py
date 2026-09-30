import numpy as np
import pytest

from miletos.phase_curve import retr_band_planck, retr_brightness_temperature, retr_rflx_pcur


def test_phase_curve_model_has_the_dayside_flux_at_eclipse_and_the_transit_depth():
    peri = 1.2749  # [day]
    time = np.linspace(-0.5 * peri, 1.5 * peri, 20001)  # [day]
    rflx = retr_rflx_pcur(time, 0., peri, 0.12, 0.3, 0.0, np.zeros(2), 5e-4, 1e-4, 0., 0.)
    # just before secondary eclipse the planet shows its phase-dependent flux, which vanishes at mid-eclipse
    ingress = 0.5 * peri - 0.08
    anglphas = 2. * np.pi * ingress / peri  # [rad]
    fluxcomp = 1e-4 + (5e-4 - 1e-4) * 0.5 * (1. - np.cos(anglphas))
    assert rflx[np.argmin(np.abs(time - ingress))] - rflx[np.argmin(np.abs(time - 0.5 * peri))] == \
        pytest.approx(fluxcomp, rel=1e-3)
    # a uniform stellar disk loses the planet's area fraction at mid-transit, while the nightside still shines
    assert 1. - rflx[np.argmin(np.abs(time))] == pytest.approx(0.12**2 - 1e-4, rel=1e-3)


def test_brightness_temperature_inverts_the_band_flux_ratio():
    wavelength = np.linspace(600e-9, 1000e-9, 200)  # [m]
    response = np.ones_like(wavelength)
    ratiflux = 0.12**2 * retr_band_planck(2500., wavelength, response)[0] / retr_band_planck(6500., wavelength, response)[0]
    assert retr_brightness_temperature(ratiflux, 0.12, 6500., wavelength, response) == pytest.approx(2500., rel=1e-6)
    assert retr_brightness_temperature(-1e-5, 0.12, 6500., wavelength, response) == 0.
