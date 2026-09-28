import os
from pathlib import Path
import subprocess
import sys

import matplotlib.image as mpimg
import pytest


REPOSITORY_PATH = Path(__file__).parents[1]
EXAMPLE_PATHS = sorted((REPOSITORY_PATH / 'examples').glob('*.py'))


@pytest.mark.parametrize('example_path', EXAMPLE_PATHS, ids=lambda path: path.name)
def test_example_cli_help(example_path):
    environment = os.environ.copy()
    environment['MILETOS_PATH'] = str(REPOSITORY_PATH)

    completed = subprocess.run(
        [sys.executable, str(example_path), '--help'],
        cwd=REPOSITORY_PATH,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert 'usage:' in completed.stdout


def test_simulated_transit_example_runs(tmp_path):
    environment = os.environ.copy()
    environment['MILETOS_PATH'] = str(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_PATH / 'examples' / 'simulated_transit_diagnostic.py'),
            '--typefileplot',
            'png',
        ],
        cwd=REPOSITORY_PATH,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    output_path = tmp_path / 'visuals' / 'simulated_transit_diagnostic.png'
    assert completed.returncode == 0, completed.stderr
    assert output_path.is_file()
    assert f'Writing to {output_path}...' in completed.stdout


def test_all_examples_run_and_produce_plots(tmp_path):
    environment = os.environ.copy()
    environment['MILETOS_PATH'] = str(tmp_path)

    completed = subprocess.run(
        [sys.executable, str(REPOSITORY_PATH / 'examples' / 'run_all.py')],
        cwd=REPOSITORY_PATH,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    expected_names = (
        'simulated_transit_diagnostic.png',
        'target_visibility_toi-1233.png',
        'simulated_wasp39_jwst_diagnostic.png',
        'example_catalog_transit.png',
    )
    for name in expected_names:
        output_path = tmp_path / 'visuals' / name
        image = mpimg.imread(output_path)
        assert output_path.is_file()
        assert image.shape[0] > 100
        assert image.shape[1] > 100
        assert image[..., :3].min() < 0.8