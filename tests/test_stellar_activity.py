import numpy as np
import pytest

from miletos import stellar_activity as analysis


@pytest.mark.parametrize("evolving_spots", [False, True])
def test_parameter_blocks_do_not_overlap_spots_and_flares(evolving_spots):
    spots = [[10.0, 20.0, 0.2], [-30.0, 40.0, 0.1]]
    if evolving_spots:
        spots = [spots[0] + [1.0, 2.0], spots[1] + [3.0, 4.0]]
    flares = np.array([[5.0, 0.1, 0.2, 0.3], [6.0, 0.2, 0.3, 0.4]])
    parameters = np.concatenate(([0.3, 0.2, 10.0, 90.0, 0.2, 0.01],
                                 np.asarray(spots).ravel(), flares.ravel()))

    result = analysis.parse_spot_flare_parameters(
        parameters, number_spots=2, number_flares=2, evolving_spots=evolving_spots,
    )

    np.testing.assert_allclose(result["spot_radius_ratios"], [0.2, 0.1])
    np.testing.assert_allclose(result["flare_peak_times_days"], [5.0, 6.0])
    np.testing.assert_allclose(result["flare_decay_times_days"], [0.3, 0.4])
    if evolving_spots:
        np.testing.assert_allclose(result["spot_widths_days"], [2.0, 4.0])
    assert analysis.log_prior_spot_flare(parameters, number_spots=2, number_flares=2,
                                        evolving_spots=evolving_spots) == 0.0


def test_zero_spots_with_flare_and_gaussian_likelihood():
    time_days = np.linspace(-1.0, 1.0, 101)  # [day]
    parameters = [0.3, 0.3, 10.0, 90.0, 0.0, 0.01, 0.0, 0.1, 0.1, 0.2]
    model_parameters = analysis.parse_spot_flare_parameters(parameters, number_spots=0, number_flares=1)
    model_flux = analysis.evaluate_spot_flare_model(time_days, **model_parameters)
    uncertainty = np.full(time_days.size, 0.01)

    likelihood = analysis.log_likelihood_spot_flare(
        parameters, time_days, model_flux + uncertainty, uncertainty,
        number_spots=0, number_flares=1,
    )

    assert likelihood == pytest.approx(-0.5 * time_days.size)


def test_ordering_prior_rejects_ascending_spot_radii():
    parameters = [0.3, 0.3, 10.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 90.0, 0.2]
    assert analysis.log_prior_spot_flare(parameters, number_spots=2) == -np.inf


def test_local_fit_recovers_baseline_flux_offset():
    time_days = np.linspace(0.0, 10.0, 31)  # [day]
    flux = np.full(time_days.size, 1.02)
    uncertainty = np.full(time_days.size, 0.001)
    initial = [0.3, 0.3, 10.0, 90.0, 0.0, 0.0]
    bounds = ([0.0, 0.0, 1.0, 0.0, -0.5, -0.05], [1.0, 1.0, 20.0, 180.0, 0.5, 0.05])

    result = analysis.fit_spot_flare_light_curve(
        time_days, flux, uncertainty, initial, number_spots=0, bounds=bounds,
    )

    assert result["optimizer"].success
    assert result["parameters"][5] == pytest.approx(0.02, abs=1e-7)
    np.testing.assert_allclose(result["model_flux"], flux, atol=1e-7)


def test_likelihood_calls_ephesos_with_parsed_parameters(monkeypatch):
    calls = []

    def model(time_days, **parameters):
        calls.append(parameters)
        return np.ones_like(time_days)

    monkeypatch.setattr(analysis, "evaluate_spot_flare_model", model)
    result = analysis.log_likelihood_spot_flare(
        [0.3, 0.3, 10.0, 90.0, 0.2, 0.0, 0.0, 20.0, 0.1],
        [0.0, 1.0], [1.0, 1.0], [0.01, 0.01], number_spots=1,
    )

    assert result == 0.0
    np.testing.assert_allclose(calls[0]["spot_longitudes_degrees"], [20.0])


def test_likelihood_rejects_nonpositive_uncertainty():
    with pytest.raises(ValueError, match="uncertainties"):
        analysis.log_likelihood_spot_flare(
            [0.3, 0.3, 10.0, 90.0, 0.0, 0.0], [0.0], [1.0], [0.0], number_spots=0,
        )


def test_variable_star_selection_uses_sector_40_inclusively():
    import pandas as pd

    catalog = pd.DataFrame({"RAJ2000": [1.0, 2.0, 3.0], "Period": [4.0, 5.0, 6.0]})
    coverage = np.zeros((3, 41), dtype=bool)
    coverage[0, 38] = True
    coverage[1, 39] = True
    coverage[2, 40] = True

    result = analysis.select_observed_variable_stars(catalog, coverage)

    assert result.index.tolist() == [1, 2]
    assert result["Period"].tolist() == [5.0, 6.0]


def test_variable_star_selection_rejects_unaligned_catalog():
    import pandas as pd

    with pytest.raises(ValueError, match="one row"):
        analysis.select_observed_variable_stars(pd.DataFrame({"target": [1]}), np.zeros((2, 40)))


def test_seeded_flare_example_fits_and_renders(tmp_path):
    import importlib.util
    from pathlib import Path
    import matplotlib.image as mpimg

    path = Path(__file__).parents[1] / "examples" / "stellar_activity" / "run.py"
    specification = importlib.util.spec_from_file_location("stellar_activity_example", path)
    example = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(example)
    output_path = tmp_path / "stellar_activity.png"

    result = example.run_example(output_path)

    assert result["optimizer"].success
    assert result["parameters"][6] == pytest.approx(1.5, abs=0.03)
    assert result["parameters"][7] == pytest.approx(0.03, rel=0.1)
    assert np.isfinite(result["model_flux"]).all()
    image = mpimg.imread(output_path)
    assert image[..., :3].min() < 0.8