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