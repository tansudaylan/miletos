import numpy as np
import pytest

from miletos.main import retr_llik_albbepsi, setp_input_companion_priors


class DummyData:
    pass


def test_albedo_likelihood_has_equal_hemispheres_at_full_redistribution():
    state = DummyData()
    state.gmeatmptequi = 1000.  # [K]
    state.gmeatmptdayy = 1000. * 0.8**0.25  # [K]
    state.gmeatmptnigh = state.gmeatmptdayy  # [K]
    state.gstdtmptdayy = 20.  # [K]
    state.gstdtmptnigh = 20.  # [K]
    state.gmeapsii = 0.8**0.25
    state.gstdpsii = 0.1

    assert np.isclose(retr_llik_albbepsi(np.array([0.2, 0.5, 1.]), state), 0.)
    assert retr_llik_albbepsi(np.array([0.2, 0.5, 0.5]), state) < 0.


def make_input():
    gdat = DummyData()
    gdat.nomipara = DummyData()
    gdat.indxband = np.arange(1)
    gdat.rratcompprio = [0.01638, 0.02134, 0.02805, 0.0323]
    gdat.rsmacompprio = [0.0895, 0.0647, 0.0375, 0.03043]
    gdat.epocmtracompprio = [2458572.1128, 2458572.3949, 2458571.3368, 2458586.5677]
    gdat.pericompprio = [3.79523, 6.20370, 14.17555, 19.5917]
    gdat.cosicompprio = [0.037, 0.022, 0.0136, 0.0118]
    return gdat


def test_input_companion_priors_populate_nominal_model():
    gdat = make_input()

    setp_input_companion_priors(gdat)

    assert np.allclose(gdat.nomipara.pericomp, gdat.pericompprio)
    assert np.allclose(gdat.nomipara.epocmtracomp, gdat.epocmtracompprio)
    assert np.allclose(gdat.nomipara.rratcomp[0], gdat.rratcompprio)
    assert np.allclose(
        gdat.nomipara.depttrancomp,
        1e3 * np.asarray(gdat.rratcompprio) ** 2,
    )
    assert gdat.nomipara.duratrantotlcomp.shape == (4,)


def test_input_companion_priors_require_equal_lengths():
    gdat = make_input()
    gdat.cosicompprio = gdat.cosicompprio[:-1]

    with pytest.raises(ValueError, match='equal lengths'):
        setp_input_companion_priors(gdat)


@pytest.mark.parametrize('name', ['rrat', 'rsma', 'epocmtra', 'peri', 'cosi'])
@pytest.mark.parametrize('kind', ['masked', 'complex'])
def test_input_companion_priors_reject_hidden_or_complex_values_before_model_changes(monkeypatch, name, kind):
    import importlib

    model = importlib.import_module('miletos.main')
    state = make_input()
    field = name + 'compprio'
    original = np.asarray(getattr(state, field), dtype=float)
    if kind == 'masked':
        invalid = np.ma.array(original, mask=[False, True, False, False])
    else:
        invalid = original.astype(complex)
        invalid[0] += 0.1j
    setattr(state, field, invalid)
    original_data = np.ma.getdata(invalid).copy()
    original_mask = np.ma.getmaskarray(invalid).copy()
    state.nomipara.existing = 'preserve'
    monkeypatch.setattr(model.tdpy, 'retr_duratrantotl',
                        lambda *args: pytest.fail('Invalid priors must be rejected before derived calculations'))

    with pytest.raises(ValueError, match='unmasked real'):
        setp_input_companion_priors(state)

    assert vars(state.nomipara) == {'existing': 'preserve'}
    np.testing.assert_array_equal(np.ma.getdata(invalid), original_data)
    np.testing.assert_array_equal(np.ma.getmaskarray(invalid), original_mask)


@pytest.mark.parametrize('name', ['rrat', 'rsma', 'epocmtra', 'peri', 'cosi'])
@pytest.mark.parametrize('kind', ['nan', 'positive_infinity', 'negative_infinity', 'matrix'])
def test_input_companion_priors_reject_nonfinite_or_nonscalar_axes_before_changes(monkeypatch, name, kind):
    import importlib

    model = importlib.import_module('miletos.main')
    state = make_input()
    field = name + 'compprio'
    invalid = np.asarray(getattr(state, field), dtype=float)
    if kind == 'matrix':
        invalid = invalid.reshape(2, 2)
    else:
        invalid[0] = {'nan': np.nan, 'positive_infinity': np.inf,
                      'negative_infinity': -np.inf}[kind]
    original = invalid.copy()
    setattr(state, field, invalid)
    state.nomipara.existing = 'preserve'
    monkeypatch.setattr(model.tdpy, 'retr_duratrantotl',
                        lambda *args: pytest.fail('Invalid priors reached derived calculations'))

    with pytest.raises(ValueError, match='one-dimensional|finite'):
        setp_input_companion_priors(state)

    assert vars(state.nomipara) == {'existing': 'preserve'}
    np.testing.assert_array_equal(invalid, original)


@pytest.mark.parametrize('container', ['list', 'array', 'unmasked'])
def test_input_companion_priors_preserve_real_container_results(container):
    expected_state = make_input()
    setp_input_companion_priors(expected_state)
    state = make_input()
    state.indxband = np.arange(2)
    originals = {}
    for name in ('rrat', 'rsma', 'epocmtra', 'peri', 'cosi'):
        field = name + 'compprio'
        values = np.asarray(getattr(state, field))
        supplied = (values.tolist() if container == 'list' else
                    np.ma.array(values, mask=False) if container == 'unmasked' else values)
        setattr(state, field, supplied)
        originals[field] = np.asarray(supplied).copy()

    setp_input_companion_priors(state)

    for field in ('rsmacomp', 'epocmtracomp', 'pericomp', 'cosicomp',
                  'depttrancomp', 'duratrantotlcomp'):
        np.testing.assert_array_equal(getattr(state.nomipara, field),
                                      getattr(expected_state.nomipara, field))
    for radius_ratios in state.nomipara.rratcomp:
        np.testing.assert_array_equal(radius_ratios, expected_state.nomipara.rratcomp[0])
    assert not np.shares_memory(*state.nomipara.rratcomp)
    for field, values in originals.items():
        np.testing.assert_array_equal(getattr(state, field), values)