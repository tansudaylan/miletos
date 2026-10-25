import numpy as np
import pytest

import ephesos
import miletos
from miletos.signatures import compute_photometric_signatures


def test_public_miletos_signature_api_uses_ephesos_predictions():
    period_days = np.array([0.3, 30.0])[:, None]  # [day]
    companion_mass_solar = np.array([5.0, 180.0])[None, :]  # [solar mass]

    assert miletos.compute_photometric_signatures is compute_photometric_signatures
    signatures = miletos.compute_photometric_signatures(
        period_days, companion_mass_solar
    )
    expected = ephesos.evaluate_compact_object_signatures(
        period_days, companion_mass_solar
    )
    for name in expected:
        np.testing.assert_array_equal(signatures[name], expected[name])


def test_miletos_derives_compact_object_features():
    features = miletos.derive_compact_object_features(1.0, 10.0, 5.0, 1.0)

    assert set(features) == {
        "amplbeammodl",
        "ampellimodl",
        "amplslenmodl",
        "duratrantotlmodl",
        "smaxmodl",
        "radischw",
    }
    assert all(np.isfinite(values).all() for values in features.values())
    signatures = miletos.compute_photometric_signatures(10.0, 5.0)
    np.testing.assert_allclose(features["amplbeammodl"], [signatures["beaming"]])
    np.testing.assert_allclose(features["ampellimodl"], [signatures["ellipsoidal"]])
    np.testing.assert_allclose(features["amplslenmodl"], [3.015272], rtol=1e-6)


def test_compact_object_features_use_density_from_the_same_stellar_mass_and_radius():
    features = miletos.derive_compact_object_features([1.0, 2.0], 10.0, 5.0, 1.0)
    expected = ephesos.evaluate_compact_object_signatures(
        10.0, 5.0, stellar_radius_solar=[1.0, 2.0], stellar_mass_solar=1.0,
        stellar_density_cgs=[1.41, 1.41 / 8.0],
    )
    np.testing.assert_allclose(features["ampellimodl"], expected["ellipsoidal"])
    assert features["ampellimodl"][1] == pytest.approx(8.0 * features["ampellimodl"][0])
    np.testing.assert_array_equal(features["amplbeammodl"], expected["beaming"])


@pytest.mark.parametrize("masked_index", range(4))
def test_compact_object_features_reject_masked_physical_inputs(masked_index):
    arguments = [1.0, 10.0, 5.0, 1.0]
    arguments[masked_index] = np.ma.array(arguments[masked_index], mask=True)
    with pytest.raises(ValueError, match="measured"):
        miletos.derive_compact_object_features(*arguments)


@pytest.mark.parametrize("index,value", [(0, 0.0), (0, np.nan), (1, -1.0), (2, np.inf), (3, 0.0)])
def test_compact_object_features_reject_invalid_inputs_before_density_calculation(index, value):
    arguments = [1.0, 10.0, 5.0, 1.0]
    arguments[index] = value
    with np.errstate(all="raise"), pytest.raises(ValueError, match="finite and positive"):
        miletos.derive_compact_object_features(*arguments)