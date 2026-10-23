import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "TIC38455598_exocomet"
    / "run.py"
)
SPEC = importlib.util.spec_from_file_location(
    "miletos_tic38455598_exocomet_example", EXAMPLE_PATH
)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


def test_example_figures_use_its_visuals_directory():
    assert EXAMPLE.DEFAULT_VISUAL_DIRECTORY == EXAMPLE_PATH.parent / "visuals"


def test_readme_uses_rendered_faintstar_and_ephesos_figures():
    repository_root = EXAMPLE_PATH.parents[2]
    readme_path = repository_root / "README.md"
    print(f"Reading from {readme_path}...")
    readme = readme_path.read_text(encoding="utf-8")
    for filename, alt_text in (
        (
            "faintstar_sector62.png",
            "Miletos FaintStar BLS vetting of TIC 38455598.1 in TESS Sector 62",
        ),
        (
            "ephesos_exocomet_model.png",
            "Ephesos opaque-disk and trailing dust-tail fits to the phase-folded TIC 38455598.1 event",
        ),
    ):
        relative_path = Path("examples/TIC38455598_exocomet/visuals") / filename
        figure_path = repository_root / relative_path
        assert f"![{alt_text}]({relative_path.as_posix()})" in readme
        with Image.open(figure_path) as image:
            assert image.width >= 1000 and image.height >= 500
            assert any(minimum < maximum for minimum, maximum in image.getextrema())


def test_ephesos_model_comparison_uses_gaussian_objective_and_bic(monkeypatch):
    losses = []

    def initial_parameters_only(residual_function, *, x0, **kwargs):
        losses.append(kwargs["loss"])
        residual_function(np.asarray(x0))
        return SimpleNamespace(x=np.asarray(x0))

    monkeypatch.setattr(EXAMPLE, "least_squares", initial_parameters_only)
    phase_days = np.linspace(-0.1, 0.1, 13)
    flux = np.ones(phase_days.size)
    uncertainty = np.full(phase_days.size, 0.001)

    fit = EXAMPLE._fit_ephesos_models(
        phase_days,
        flux,
        uncertainty,
        period_days=4.1855,
        stellar_mass_solar=EXAMPLE.STELLAR_MASS_SOLAR,
        stellar_radius_solar=EXAMPLE.STELLAR_RADIUS_SOLAR,
    )

    assert losses == ["linear", "linear"]
    assert np.isfinite(fit["disk_bic"])
    assert np.isfinite(fit["comet_bic"])
    assert fit["disk_bic"] == fit["disk_chi_square"] + 2 * np.log(flux.size)
    assert fit["comet_bic"] == fit["comet_chi_square"] + 6 * np.log(flux.size)