import os
from pathlib import Path
import subprocess
import sys

import matplotlib.image as mpimg
import pytest


REPOSITORY_PATH = Path(__file__).parents[1]
EXAMPLE_PATHS = [REPOSITORY_PATH / 'examples' / 'run_all.py']
EXAMPLE_PATHS.extend(sorted((REPOSITORY_PATH / 'examples').glob('*/run.py')))


def example_id(path):
    return path.stem if path.parent.name == 'examples' else path.parent.name


def test_each_example_owns_subfolder():
    root_scripts = set((REPOSITORY_PATH / 'examples').glob('*.py'))
    assert root_scripts == {REPOSITORY_PATH / 'examples' / 'run_all.py'}
    assert len(EXAMPLE_PATHS) == 6


@pytest.mark.parametrize('example_path', EXAMPLE_PATHS, ids=example_id)
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
            str(REPOSITORY_PATH / 'examples' / 'simulated_transit' / 'run.py'),
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

    output_path = (
        tmp_path
        / 'examples'
        / 'simulated_transit'
        / 'visuals'
        / 'simulated_transit_diagnostic.png'
    )
    assert completed.returncode == 0, completed.stderr
    assert output_path.is_file()
    assert f'Writing to {output_path}...' in completed.stdout


def test_all_examples_run_and_produce_plots(tmp_path):
    environment = os.environ.copy()
    environment['MILETOS_PATH'] = str(tmp_path)

    completed = subprocess.run(
        [sys.executable, str(REPOSITORY_PATH / 'examples' / 'run_all.py'), '--quick'],
        cwd=REPOSITORY_PATH,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert 'Reproduced 344 published wavelength bins' in completed.stdout
    assert 'reduced chi-squared = 1.10' in completed.stdout
    expected_names = (
        'examples/simulated_transit/visuals/simulated_transit_diagnostic.png',
        'examples/target_visibility/visuals/target_visibility_toi-1233.png',
        'examples/WASP-39b/visuals/wasp39_ers_g395h_transmission_spectrum.png',
        'examples/WASP-39b/visuals/wasp39_ers_g395h_white_light.png',
        'examples/WASP-39b/visuals/wasp39_ers_g395h_detector_motion.png',
        'examples/WASP-39b/visuals/wasp39_ers_g395h_spectroscopic_detrending.png',
        'examples/WASP-39b/visuals/wasp39_ers_g395h_corrected_light_curve_map.png',
        'examples/WASP-39b/visuals/wasp39_ers_g395h_light_curve_precision.png',
        'examples/catalog/visuals/example_catalog_transit.png',
    )
    for name in expected_names:
        output_path = tmp_path / name
        image = mpimg.imread(output_path)
        assert output_path.is_file()
        assert image.shape[0] > 100
        assert image.shape[1] > 100
        assert image[..., :3].min() < 0.8


def test_run_all_lists_every_catalog_configuration():
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY_PATH / 'examples' / 'run_all.py'), '--list'],
        cwd=REPOSITORY_PATH,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    tree = __import__('ast').parse(
        (REPOSITORY_PATH / 'examples' / 'catalog' / 'run.py').read_text()
    )
    expected = {
        node.name for node in tree.body
        if isinstance(node, __import__('ast').FunctionDef) and node.name.startswith('cnfg_')
    }
    listed = {line.split()[0] for line in completed.stdout.splitlines()}
    assert completed.returncode == 0, completed.stderr
    assert listed == expected
    assert len(completed.stdout.splitlines()) == 45