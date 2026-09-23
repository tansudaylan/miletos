import builtins
import importlib
import sys


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