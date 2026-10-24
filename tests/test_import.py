import builtins
import importlib
import sys


def test_import_miletos_does_not_eagerly_import_main():
    sys.modules.pop('miletos.main', None)
    sys.modules.pop('miletos', None)

    package = importlib.import_module('miletos')

    assert 'miletos.main' not in sys.modules
    assert package.__name__ == 'miletos'


def test_import_miletos_without_optional_lygos(monkeypatch):
    """Core time-series workflows import without the optional photometry backend."""

    real_import = builtins.__import__

    def reject_lygos(name, *args, **kwargs):
        if name == 'lygos' or name.startswith('lygos.'):
            raise ModuleNotFoundError("No module named 'lygos'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', reject_lygos)
    sys.modules.pop('miletos', None)
    sys.modules.pop('miletos.main', None)

    package = importlib.import_module('miletos')

    assert hasattr(package, 'init')


def test_pytest_configuration_collects_duplicate_module_names(tmp_path):
    from pathlib import Path
    import subprocess

    files = []
    for name in ('first', 'second'):
        directory = tmp_path / name
        directory.mkdir()
        path = directory / 'test_shared_name.py'
        path.write_text('def test_collected():\n    pass\n')
        files.append(str(path))
    config = Path(__file__).parents[1] / 'pyproject.toml'
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', '--collect-only', '-c', str(config), *files],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '2 tests collected' in result.stdout