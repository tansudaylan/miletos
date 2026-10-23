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
    Path('examples/TOI-1233/Daylan2021.ipynb'): 'fit_daylan2021a_transits(',
    Path('examples/TOI-1233/Daylan2021 2.ipynb'): 'main.init(',
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
    Path('examples/WASP-39b/WASP39ERS.ipynb'): {'matplotlib', 'numpy', 'pandas', 'scipy'},
}


def example_id(path):
    return path.stem if path.parent.name == 'examples' else path.parent.name


def test_each_example_owns_subfolder():
    root_scripts = set((REPOSITORY_PATH / 'examples').glob('*.py'))
    assert root_scripts == {REPOSITORY_PATH / 'examples' / 'run_all.py'}
    assert len(EXAMPLE_PATHS) == 12


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


def test_daylan_target_notebook_reproduces_published_posterior():
    path = REPOSITORY_PATH / 'examples' / 'TOI-1233' / 'Daylan2021.ipynb'
    print(f'Reading from {path}...')
    notebook = json.loads(path.read_text())
    source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code')

    assert 'fit_daylan2021a_transits(' in source
    assert 'retr_summary_daylan2021a(' in source
    assert 'plot_daylan2021a_transits(' in source
    assert 'figure_path.relative_to(get_repository_path())' in source
    assert 'run_toi1233_observation(' not in source
    assert 'from pcat' not in source


def test_daylan2021a_verbose_file_messages_use_relative_paths(tmp_path, monkeypatch):
    import numpy as np

    from miletos import daylan2021a

    data_path = tmp_path / 'data'
    repository_path = tmp_path / 'repository'
    time = daylan2021a.EPOC_PUBL[0] + np.linspace(-0.2, 0.2, 100)
    photometry = np.column_stack((time, np.ones(time.size), np.full(time.size, 1e-4)))
    for sector in (10, 11):
        path = data_path / 'TOI-1233' / 'data' / f'RelativeFlux_DataCube_Detrended_TESS_Sector{sector}_2min.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, photometry, delimiter=',')

    posterior_path = repository_path / 'examples' / 'TOI-1233' / 'data' / 'daylan2021a_transit_posterior.npz'
    posterior_path.parent.mkdir(parents=True)
    np.savez(posterior_path, samples=np.zeros((1, 22)), radi=np.ones((1, 4)))
    messages = []
    monkeypatch.setattr(daylan2021a, 'get_data_path', lambda: data_path)
    monkeypatch.setattr(daylan2021a, 'get_repository_path', lambda: repository_path)
    monkeypatch.setattr(daylan2021a, 'print', messages.append)

    daylan2021a.fit_daylan2021a_transits(boolreus=True)

    assert len(messages) == 3
    assert messages[-1] == 'Reading from examples/TOI-1233/data/daylan2021a_transit_posterior.npz...'
    assert all(str(tmp_path) not in message for message in messages)


def test_daylan2021a_summary_reports_published_planet_values():
    import numpy as np

    from miletos.daylan2021a import NAMEPLAN, PERI_PUBL, RADI_PUBL, retr_summary_daylan2021a

    samples = np.zeros((16, 22))
    for index, period in enumerate(PERI_PUBL):
        samples[:, 5 * index + 1] = period
    radii = np.tile(RADI_PUBL[:, 0], (samples.shape[0], 1))

    summary = retr_summary_daylan2021a(samples, radii)

    assert len(summary) == len(NAMEPLAN)
    assert all('published' in line for line in summary)
    assert all(
        f'{period:.5f}' in line
        for period, line in zip(PERI_PUBL, summary)
    )


def test_daylan2021b_notebook_runs_and_summarizes_phase_curve_reproduction():
    path = REPOSITORY_PATH / 'examples' / 'WASP-121b' / 'Daylan2021b.ipynb'
    print(f'Reading from {path}...')
    notebook = json.loads(path.read_text())
    source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code')

    assert 'fit_daylan2021b_phase_curve(' in source
    assert 'retr_summary_daylan2021b(' in source
    assert "daylan2021b_phase_curve.png" in source
    assert "daylan2021b_brightness_temperatures.png" in source
    assert 'run_posterior_sampling = False' not in source


def test_published_reproduction_figures_are_linked_from_readme():
    figures = (
        'examples/WASP-39b/visuals/wasp39_ers_g395h_transmission_spectrum.png',
        'examples/WASP-39b/visuals/wasp39_ers_g395h_white_light.png',
        'examples/WASP-39b/visuals/wasp39_ers_g395h_detector_motion.png',
        'examples/WASP-39b/visuals/wasp39_ers_g395h_spectroscopic_detrending.png',
        'examples/WASP-39b/visuals/wasp39_ers_g395h_light_curve_precision.png',
        'examples/WASP-39b/visuals/wasp39_ers_g395h_corrected_light_curve_map.png',
        'examples/TOI-1233/visuals/daylan2021a_published_comparison.png',
        'examples/TOI-1233/visuals/daylan2021a_phase_folded_transits.png',
        'examples/WASP-121b/visuals/daylan2021b_phase_curve.png',
        'examples/WASP-121b/visuals/daylan2021b_brightness_temperatures.png',
    )
    readme = (REPOSITORY_PATH / 'README.md').read_text()

    for relative_path in figures:
        figure_path = REPOSITORY_PATH / relative_path
        assert figure_path.is_file(), relative_path
        assert figure_path.stat().st_size > 1000, relative_path
        assert relative_path in readme, relative_path


def _example_environment(path_override=None):
    environment = os.environ.copy()
    if path_override is not None:
        environment['MILETOS_PATH'] = str(path_override)
    pythonpath_parts = [
        str(REPOSITORY_PATH),
        *(str(path) for path in REPOSITORY_PATH.parent.iterdir() if path.is_dir()),
    ]
    if 'PYTHONPATH' in environment:
        pythonpath_parts.append(environment['PYTHONPATH'])
    environment['PYTHONPATH'] = os.pathsep.join(pythonpath_parts)
    return environment


@pytest.mark.parametrize('example_path', EXAMPLE_PATHS, ids=example_id)
def test_example_cli_help(example_path):
    environment = _example_environment(REPOSITORY_PATH)

    completed = subprocess.run(
        [sys.executable, str(example_path), '--help'],
        cwd=REPOSITORY_PATH,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert 'usage:' in completed.stdout


def test_simulated_transit_example_runs(tmp_path):
    environment = _example_environment(tmp_path)

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
    environment = _example_environment(tmp_path)

    completed = subprocess.run(
        [sys.executable, str(REPOSITORY_PATH / 'examples' / 'run_all.py'), '--quick'],
        cwd=REPOSITORY_PATH,
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert 'Reproduced 344 published wavelength bins' in completed.stdout
    assert 'reduced chi-squared = 1.10' in completed.stdout
    expected_names = (
        'examples/stellar_activity/visuals/stellar_activity.png',
        'examples/simulated_transit/visuals/simulated_transit_diagnostic.png',
        'examples/nirspec_isotope_retrieval/visuals/nirspec_isotope_injection_recovery.png',
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
    environment = _example_environment()
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY_PATH / 'examples' / 'run_all.py'), '--list'],
        cwd=REPOSITORY_PATH,
        env=environment,
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