"""Miletos public package interface."""

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
	"""Load legacy root-level workflow attributes only when requested."""

	main_module = import_module('.main', __name__)
	if name == 'main':
		return main_module
	try:
		value = getattr(main_module, name)
	except AttributeError as exception:
		raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from exception
	globals()[name] = value
	return value
