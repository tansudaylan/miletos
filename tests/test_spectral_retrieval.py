import numpy as np
import pytest

from miletos.retrieval import _spectral_log_likelihood, sample_nirspec_retrieval


@pytest.fixture
def retrieval_inputs():
    return {
        "wavelength_microns": np.array([4.0, 4.1]),  # [micron]
        "observed_depth_ppm": np.array([20000.0, 20010.0]),  # [ppm]
        "covariance_ppm2": np.eye(2),  # [ppm^2]
        "forward_model": lambda parameters, wavelengths: 20000.0 + np.full(len(wavelengths), parameters[0]),
        "parameter_names": ("depth_offset_ppm",),
        "prior_types": ("self",),
        "prior_minima": np.array([-10.0]),  # [ppm]
        "prior_maxima": np.array([10.0]),  # [ppm]
        "initial_values": np.array([0.0]),  # [ppm]
        "proposal_scales": np.array([0.1]),
        "prior_means": np.array([0.0]),  # [ppm]
        "prior_stdvs": np.array([1.0]),  # [ppm]
    }


@pytest.mark.parametrize("name", [
    "wavelength_microns", "observed_depth_ppm", "covariance_ppm2",
    "prior_minima", "prior_maxima", "initial_values", "proposal_scales",
    "prior_means", "prior_stdvs",
])
@pytest.mark.parametrize("kind", ["masked", "complex"])
def test_retrieval_rejects_missing_and_nonreal_inputs_before_sampling(monkeypatch, retrieval_inputs, name, kind):
    from unittest.mock import Mock
    import pcat.fixed

    values = retrieval_inputs[name]
    if kind == "masked":
        mask = np.zeros(values.shape, dtype=bool)
        mask.flat[0] = True
        retrieval_inputs[name] = np.ma.array(values, mask=mask)
    else:
        retrieval_inputs[name] = values.astype(complex) + 1j
    sampler = Mock(side_effect=AssertionError("Invalid inputs reached PCAT"))
    monkeypatch.setattr(pcat.fixed, "sample_fixed_chains", sampler)

    with pytest.raises(ValueError, match=f"{name}.*unmasked real"):
        sample_nirspec_retrieval(**retrieval_inputs)

    sampler.assert_not_called()


@pytest.mark.parametrize("kind", ["masked", "complex", "unmasked"])
def test_spectral_likelihood_validates_forward_model_values(retrieval_inputs, kind):
    output = retrieval_inputs["observed_depth_ppm"].copy()
    if kind == "complex":
        output = output.astype(complex) + 1j
    else:
        output = np.ma.array(output, mask=[kind == "masked", False])
    original = output.copy()
    state = {
        "forward_model": lambda parameters, wavelengths: output,
        "wavelength_microns": retrieval_inputs["wavelength_microns"],
        "observed_depth_ppm": retrieval_inputs["observed_depth_ppm"],
        "covariance_cholesky": np.eye(2),
        "model_error_types": (),
    }

    actual = _spectral_log_likelihood(np.array([0.0]), state)

    assert actual == (0.0 if kind == "unmasked" else -np.inf)
    np.testing.assert_array_equal(np.ma.getdata(output), np.ma.getdata(original))
    np.testing.assert_array_equal(np.ma.getmaskarray(output), np.ma.getmaskarray(original))


def test_retrieval_preserves_fully_unmasked_containers(monkeypatch, retrieval_inputs):
    import pcat.fixed

    def sample(state, likelihood, *args, **kwargs):
        return likelihood(np.array([0.0]), state), args

    monkeypatch.setattr(pcat.fixed, "sample_fixed_chains", sample)
    expected_likelihood, expected_arguments = sample_nirspec_retrieval(**retrieval_inputs)
    containers = {
        name: np.ma.array(values, mask=False) if isinstance(values, np.ndarray) else values
        for name, values in retrieval_inputs.items()
    }

    actual_likelihood, actual_arguments = sample_nirspec_retrieval(**containers)

    assert actual_likelihood == expected_likelihood
    for actual, expected in zip(actual_arguments, expected_arguments):
        np.testing.assert_array_equal(actual, expected)
    for name, values in containers.items():
        if isinstance(values, np.ma.MaskedArray):
            np.testing.assert_array_equal(np.ma.getdata(values), retrieval_inputs[name])
            assert not np.ma.getmaskarray(values).any()


def test_retrieval_forwards_requested_proposal_scales(monkeypatch, retrieval_inputs):
    from unittest.mock import Mock
    import pcat.fixed

    sampler = Mock(return_value=(None, None))
    monkeypatch.setattr(pcat.fixed, "sample_fixed_chains", sampler)
    requested_scales = np.array([0.123])
    retrieval_inputs["proposal_scales"] = requested_scales

    sample_nirspec_retrieval(**retrieval_inputs)

    np.testing.assert_array_equal(sampler.call_args.kwargs["proposal_scales"], requested_scales)


def hdo_injection_model(parameters, wavelength_microns):
    """Return a synthetic NIRSpec spectrum with a tunable HDO-shaped feature."""
    depth_offset_ppm, log10_hdo_h2o = parameters
    water_feature = 320.0 * np.exp(-0.5 * ((wavelength_microns - 4.30) / 0.08) ** 2)
    hdo_feature = 120.0 * 10.0 ** (log10_hdo_h2o + 5.0) * np.exp(
        -0.5 * ((wavelength_microns - 4.12) / 0.035) ** 2
    )
    return 20000.0 + depth_offset_ppm + water_feature + hdo_feature


@pytest.fixture
def isotope_example(monkeypatch, tmp_path):
    from pathlib import Path
    import runpy

    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1]
                                   / "examples/nirspec_isotope_retrieval/run.py"))
    run = namespace["run_example"]
    monkeypatch.setitem(run.__globals__, "get_repository_path", lambda: tmp_path)
    return run


@pytest.mark.parametrize("time_tag", ["bad", "20260230_120000", "../20261026_050000"])
def test_isotope_example_rejects_invalid_timestamp_before_sampling(isotope_example, monkeypatch, time_tag):
    from unittest.mock import Mock

    sampler = Mock(side_effect=AssertionError("Invalid timestamp reached PCAT"))
    monkeypatch.setitem(isotope_example.__globals__, "sample_nirspec_retrieval", sampler)
    with pytest.raises(ValueError):
        isotope_example(time_tag=time_tag)
    sampler.assert_not_called()


def test_isotope_example_preserves_existing_figure_before_sampling(isotope_example, monkeypatch, tmp_path):
    from unittest.mock import Mock

    output = (tmp_path / "examples/nirspec_isotope_retrieval/visuals"
              / "nirspec_isotope_injection_recovery_20261026_050000.png")
    output.parent.mkdir(parents=True)
    output.write_bytes(b"existing figure")
    sampler = Mock(side_effect=AssertionError("Existing output reached PCAT"))
    monkeypatch.setitem(isotope_example.__globals__, "sample_nirspec_retrieval", sampler)

    with pytest.raises(FileExistsError):
        isotope_example(time_tag="20261026_050000")

    sampler.assert_not_called()
    assert output.read_bytes() == b"existing figure"


def test_spectral_likelihood_rejects_model_specific_invalid_states():
    class InvalidAtmosphereError(Exception):
        pass

    def invalid_model(parameters, wavelength_microns):
        raise InvalidAtmosphereError

    state = {
        "forward_model": invalid_model,
        "wavelength_microns": np.array([4.0, 4.1]),
        "observed_depth_ppm": np.array([20000.0, 20010.0]),
        "covariance_cholesky": np.eye(2),
        "model_error_types": (InvalidAtmosphereError,),
    }

    assert _spectral_log_likelihood(np.array([0.0]), state) == -np.inf


def test_nirspec_hdo_parameter_is_sampled_through_pcat(tmp_path):
    wavelength_microns = np.linspace(3.9, 4.55, 52)
    injected_parameters = np.array([12.0, -5.0])
    uncertainty_ppm = 12.0
    random = np.random.default_rng(188)
    observed_depth_ppm = hdo_injection_model(injected_parameters, wavelength_microns)
    observed_depth_ppm += random.normal(0.0, uncertainty_ppm, wavelength_microns.size)

    chain, log_probability = sample_nirspec_retrieval(
        wavelength_microns,
        observed_depth_ppm,
        np.diag(np.full(wavelength_microns.size, uncertainty_ppm**2)),
        hdo_injection_model,
        parameter_names=("depth_offset_ppm", "log10_hdo_h2o"),
        prior_types=("self", "self"),
        prior_minima=(-80.0, -7.0),
        prior_maxima=(80.0, -3.0),
        initial_values=(0.0, -5.5),
        proposal_scales=(0.08, 0.04),
        chain_count=1,
        retained_samples=500,
        burn_samples=500,
        seed=188,
        pathbase=tmp_path,
        booladaptstdp=False,
    )

    assert chain.shape == (1, 500, 2)
    assert log_probability.shape == (1, 500)
    assert np.isfinite(log_probability).all()
    assert np.median(chain[0, :, 1]) == pytest.approx(-5.0, abs=0.25)