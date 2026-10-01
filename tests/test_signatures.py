import numpy as np

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
        "amplslenmodl",
        "duratrantotlmodl",
        "smaxmodl",
        "radischw",
    }
    assert all(np.isfinite(values).all() for values in features.values())
    np.testing.assert_allclose(features["amplslenmodl"], [3.015272], rtol=1e-6)