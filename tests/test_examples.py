import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import runpy

import matplotlib.image as mpimg
import pytest


REPOSITORY_PATH = Path(__file__).parents[1]
EXAMPLE_PATHS = [REPOSITORY_PATH / 'examples' / 'run_all.py']
EXAMPLE_PATHS.extend(sorted((REPOSITORY_PATH / 'examples').glob('*/run.py')))
EXPECTED_NOTEBOOKS = {
    Path('examples/TOI-1233/Daylan2021.ipynb'): 'run_toi1233_observation(',
    Path('examples/TOI-1233/JointPhotometryRadialVelocity.ipynb'):
        'run_toi1233_observation(',
    Path('examples/WASP-39b/WASP39ERS.ipynb'):
        'run_wasp39_ers_g395h_reproduction(',
    Path('examples/WASP-121b/Daylan2021b.ipynb'):
        'run_daylan2021b_reproduction(',
    Path('examples/simulated_transit/SimulatedTransit.ipynb'):
        'run_simulated_transit_diagnostic(',
    Path('examples/tess_transit_search/TESS_Transit_Search.ipynb'):
        'search_tess_target(',
}
NOTEBOOK_IMPORT_ALLOWLIST = {
    Path('examples/tess_transit_search/TESS_Transit_Search.ipynb'): {'numpy', 'pandas'},
}


def example_id(path):
    return path.stem if path.parent.name == 'examples' else path.parent.name


def test_each_example_owns_subfolder():
    root_scripts = set((REPOSITORY_PATH / 'examples').glob('*.py'))
    assert root_scripts == {REPOSITORY_PATH / 'examples' / 'run_all.py'}
    assert len(EXAMPLE_PATHS) == 7


def test_example_notebooks_use_miletos_apis():
    notebook_paths = {
        path.relative_to(REPOSITORY_PATH)
        for path in (REPOSITORY_PATH / 'examples').glob('**/*.ipynb')
    }
    assert notebook_paths == set(EXPECTED_NOTEBOOKS)

    for relative_path, entry_point in EXPECTED_NOTEBOOKS.items():
        path = REPOSITORY_PATH / relative_path
        print(f'Reading from {path}...')
        notebook = json.loads(path.read_text())
        code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']

        assert notebook['nbformat'] == 4
        assert all(cell['metadata']['id'] == cell['id'] for cell in notebook['cells'])
        assert all('language' in cell['metadata'] for cell in notebook['cells'])

        source = '\n'.join('\n'.join(cell['source']) for cell in code_cells)
        tree = ast.parse(source)
        imported_modules = {
            node.names[0].name.split('.')[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom)) and node.names
        }

        assert entry_point in source
        allowed_imports = NOTEBOOK_IMPORT_ALLOWLIST.get(relative_path, set())
        assert (imported_modules - allowed_imports).isdisjoint(
            {'matplotlib', 'numpy', 'pandas', 'scipy'}
        )


def test_daylan_target_notebook_uses_supported_observational_pipeline():
    path = REPOSITORY_PATH / 'examples' / 'TOI-1233' / 'Daylan2021.ipynb'
    print(f'Reading from {path}...')
    notebook = json.loads(path.read_text())
    source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code')

    assert 'run_toi1233_observation(' in source
    assert 'fit_daylan2021a_transits(' not in source
    assert 'from pcat' not in source


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
        'examples/WASP-39b/visuals/wasp39_ers_g395h_transmission_spectrum.png',
        'examples/WASP-39b/visuals/wasp39_ers_g395h_white_light.png',
        'examples/WASP-39b/visuals/wasp39_ers_g395h_detector_motion.png',
        'examples/WASP-39b/visuals/wasp39_ers_g395h_spectroscopic_detrending.png',
        'examples/WASP-39b/visuals/wasp39_ers_g395h_corrected_light_curve_map.png',
        'examples/WASP-39b/visuals/wasp39_ers_g395h_light_curve_precision.png',
    )
    for name in expected_names:
        output_path = tmp_path / name
        image = mpimg.imread(output_path)
        assert output_path.is_file()
        assert image.shape[0] > 100
        assert image.shape[1] > 100
        assert image[..., :3].min() < 0.8


def test_run_all_lists_observational_targets():
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY_PATH / 'examples' / 'run_all.py'), '--list'],
        cwd=REPOSITORY_PATH,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    listed = completed.stdout.splitlines()
    assert completed.returncode == 0, completed.stderr
    assert listed == [
        'TOI-1233/run.py',
        'WASP-121b/run.py',
        'WD1856b/run.py',
        'TRAPPIST-1/run.py',
    ]


@pytest.mark.parametrize(
    ('script_name', 'target', 'model'),
    [
        ('WD1856b', 'WD 1856+534', 'PlanetarySystem'),
        ('TRAPPIST-1', 'TRAPPIST-1', 'PlanetarySystemWithTTVs'),
    ],
)
def test_observational_examples_use_tess_without_fitting(monkeypatch, script_name, target, model):
    import miletos

    arguments = []
    script = REPOSITORY_PATH / 'examples' / script_name / 'run.py'
    namespace = runpy.run_path(str(script))
    monkeypatch.setattr(miletos, 'init', lambda **kwargs: arguments.append(kwargs))
    monkeypatch.setattr(sys, 'argv', [str(script), '--typefileplot', 'pdf'])

    assert namespace['main']() == 0
    assert len(arguments) == 1
    assert arguments[0]['strgmast'] == target
    assert arguments[0]['listlablinst'] == [['TESS'], []]
    assert arguments[0]['liststrgtypedata'] == [['obsd'], []]
    assert arguments[0]['dictfitt']['typemodl'] == model
    assert arguments[0]['boolfitt'] is False
    assert arguments[0]['typefileplot'] == 'pdf'