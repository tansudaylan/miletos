from importlib import import_module


def load_batman():
    """Load Batman after setuptools supplies its distutils compatibility layer."""
    import_module("setuptools")
    return import_module("batman")