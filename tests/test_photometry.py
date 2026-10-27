import numpy as np
import pytest

ImageStack = pytest.importorskip('lygos.imagestack').ImageStack
from miletos import light_curve_from_image_stack


def image_stack():
    return ImageStack(np.arange(1., 13.).reshape(3, 2, 2),
                      np.array([2459000., 2459001., 2459002.]),
                      error=np.full((3, 2, 2), 2.), unit='e-/s', time_format='JD')


@pytest.mark.parametrize('normalize', [False, True])
def test_native_lygos_photometry_preserves_times_and_propagates_errors(monkeypatch, normalize):
    stack = image_stack()
    aperture = np.eye(2, dtype=bool)
    originals = [values.copy() for values in (stack.flux, stack.error, stack.time, stack.quality, aperture)]
    original_method = ImageStack.aperture_photometry
    calls = []

    def extract(self, mask):
        calls.append(mask.copy())
        return original_method(self, mask)

    monkeypatch.setattr(ImageStack, 'aperture_photometry', extract)
    result = light_curve_from_image_stack(stack, aperture, normalize=normalize)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], aperture)
    scale = 13. if normalize else 1.
    np.testing.assert_array_equal(result[:, 0], stack.time)
    np.testing.assert_allclose(result[:, 1], np.array([5., 13., 21.]) / scale)
    np.testing.assert_allclose(result[:, 2], np.sqrt(8.) / scale)
    for values, snapshot in zip((stack.flux, stack.error, stack.time, stack.quality, aperture), originals):
        np.testing.assert_array_equal(values, snapshot)


@pytest.mark.parametrize('field,value', [
    ('error', None),
    ('error', np.zeros((3, 2, 2))),
    ('error', np.full((3, 2, 2), -1.)),
    ('error', np.full((3, 2, 2), np.nan)),
    ('error', np.ones((3, 2, 1))),
    ('flux', np.full((3, 2, 2), np.inf)),
    ('flux', np.ones((3, 2, 2), dtype=complex)),
    ('flux', np.ma.array(np.ones((3, 2, 2)), mask=True)),
    ('time', np.array([1., 1., 2.])),
    ('time', np.array([3., 2., 1.])),
    ('time', np.array([1., np.nan, 3.])),
    ('quality', np.array([0, 1, 0])),
    ('aperture', np.zeros((2, 2), dtype=bool)),
    ('aperture', np.ones((3, 3), dtype=bool)),
    ('aperture', np.ones((2, 2))),
    ('aperture', np.ma.array(np.ones((2, 2), dtype=bool), mask=True)),
])
def test_invalid_image_inputs_are_rejected_without_modifying_them(field, value):
    stack = image_stack()
    aperture = np.eye(2, dtype=bool)
    if field == 'aperture':
        aperture = value
    else:
        setattr(stack, field, value)
    with pytest.raises(ValueError):
        light_curve_from_image_stack(stack, aperture)


def test_nonfinite_pixels_outside_aperture_do_not_change_selected_photometry():
    stack = image_stack()
    aperture = np.eye(2, dtype=bool)
    expected = light_curve_from_image_stack(stack, aperture)
    stack.flux[:, ~aperture] = np.nan
    stack.error[:, ~aperture] = np.nan
    np.testing.assert_array_equal(light_curve_from_image_stack(stack, aperture), expected)


def test_negative_measured_flux_is_preserved_without_normalization():
    stack = image_stack()
    stack.flux *= -1
    aperture = np.eye(2, dtype=bool)
    with pytest.raises(ValueError, match='positive median'):
        light_curve_from_image_stack(stack, aperture)
    np.testing.assert_array_equal(light_curve_from_image_stack(stack, aperture, normalize=False)[:, 1],
                                  [-5., -13., -21.])


def test_synthetic_image_to_miletos_search_recovers_injected_period():
    from miletos import generate_ultrasat_time_series, search_box_least_squares

    time, cadence, _ = generate_ultrasat_time_series(duration_days=6., exposure_seconds=300.)
    period = 1.2  # [day]
    duration = 0.08  # [day]
    epoch = time[0] + 0.3  # [JD]
    depth = 0.03
    phase_days = np.remainder(time - epoch + period / 2, period) - period / 2
    model = 1. - depth * (np.abs(phase_days) < duration / 2)
    rows, columns = np.indices((3, 3))
    profile = np.exp(-((rows - 1.)**2 + (columns - 1.)**2) / 2)
    profile /= profile.sum()
    rate = 10000.  # [e-/s]
    pixel_error = 2.  # [e-/s]
    expected = rate * model[:, None, None] * profile
    flux = expected + np.random.default_rng(17).normal(0., pixel_error, expected.shape)
    stack = ImageStack(flux, time, error=np.full_like(flux, pixel_error), unit='e-/s',
                       time_format='JD', mission='ULTRASAT',
                       metadata={'simulated': True, 'instrument_response_validated': False})
    data = light_curve_from_image_stack(stack, np.ones((3, 3), dtype=bool))
    result = search_box_least_squares(*data.T, duration_days=duration,
                                    minimum_period_days=1., maximum_period_days=1.4)
    assert abs(result['period_days'] - period) < cadence / 86400.
    resolution_allowance = 2 * depth * cadence / 86400. / duration
    assert abs(result['depth'] - depth) < resolution_allowance + 5 * result['depth_uncertainty']
    assert result['depth_uncertainty'] > 0
    assert stack.metadata['simulated'] and not stack.metadata['instrument_response_validated']