import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import ephesos


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "compact_object_signatures"
    / "run.py"
)
SPEC = importlib.util.spec_from_file_location(
    "miletos_compact_object_signatures_example", EXAMPLE_PATH
)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


def test_example_uses_ephesos_amplitudes_and_writes_figure(tmp_path):
    period_days = np.geomspace(0.3, 30.0, 200)[:, None]
    companion_mass_solar = np.array([5.0, 30.0, 180.0])[None, :]
    expected = ephesos.evaluate_compact_object_signatures(
        period_days,
        companion_mass_solar,
        stellar_radius_solar=1.0,
        stellar_mass_solar=1.0,
        stellar_density_cgs=1.41,
    )
    output_path = tmp_path / "compact_object_signatures.png"

    signatures = EXAMPLE.run_example(output_path)

    for name in expected:
        np.testing.assert_array_equal(signatures[name], expected[name])
    assert output_path.stat().st_size > 1000
    with Image.open(output_path) as image:
        assert image.width >= 1000
        assert image.height >= 300


def test_example_writes_optional_bokeh_html(tmp_path):
    pytest.importorskip("bokeh")
    html_path = tmp_path / "compact_object_signatures.html"

    EXAMPLE.run_example(
        tmp_path / "compact_object_signatures.png",
        bokeh_output_path=html_path,
    )

    assert html_path.stat().st_size > 1000