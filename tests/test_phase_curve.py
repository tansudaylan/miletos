import numpy as np
import pytest

from miletos import phase_curve
from miletos.phase_curve import retr_band_planck, retr_brightness_temperature, retr_rflx_pcur


def test_tess_response_skips_whitespace_only_header_lines(tmp_path, monkeypatch):
    response_path = tmp_path / 'tess-response-function-v2.0.csv'
    response_path.write_text('# Header\n \t\n# Wavelength, response\n300,0\n302,1\n')
    monkeypatch.setattr(phase_curve, 'get_data_path', lambda: tmp_path)

    wavelength, response = phase_curve.retr_tess_response()

    assert wavelength == pytest.approx([300e-9, 302e-9])
    assert response == pytest.approx([0.0, 1.0])


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


@pytest.mark.parametrize('temperatures', [2500.0, [50.0, 2500.0, 6500.0, 20000.0]])
def test_band_planck_reuses_shared_spectrum_and_preserves_photon_normalization(monkeypatch, temperatures):
    from unittest.mock import Mock
    from tdpy.spectral import blackbody_flux_wavelength

    spectrum = Mock(wraps=blackbody_flux_wavelength)
    monkeypatch.setattr(phase_curve, 'blackbody_flux_wavelength', spectrum)
    wavelength = np.array([600e-9, 650e-9, 770e-9, 850e-9, 1000e-9])  # [m]
    response = np.array([0.1, 0.5, 1.0, 0.9, 0.2])
    temperature_array = np.atleast_1d(temperatures)  # [K]
    exponent = phase_curve.PLANCK * phase_curve.LIGHT_SPEED / (
        wavelength[None, :] * phase_curve.BOLTZMANN * temperature_array[:, None]
    )
    expected = np.trapezoid(
        wavelength[None, :]**-4 / np.expm1(exponent) * response[None, :], wavelength, axis=1,
    )
    actual = retr_band_planck(temperatures, wavelength, response)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    assert actual.shape == temperature_array.shape
    spectrum.assert_called_once()
    np.testing.assert_array_equal(spectrum.call_args.args[1], wavelength[None, :] * 1e6)


def test_brightness_temperature_inverts_the_band_flux_ratio():
    wavelength = np.linspace(600e-9, 1000e-9, 200)  # [m]
    response = np.ones_like(wavelength)
    ratiflux = 0.12**2 * retr_band_planck(2500., wavelength, response)[0] / retr_band_planck(6500., wavelength, response)[0]
    assert retr_brightness_temperature(ratiflux, 0.12, 6500., wavelength, response) == pytest.approx(2500., rel=1e-6)
    assert retr_brightness_temperature(-1e-5, 0.12, 6500., wavelength, response) == 0.
