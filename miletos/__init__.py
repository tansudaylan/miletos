"""Miletos public package interface."""

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
	"""Load legacy root-level workflow attributes only when requested."""

	if name == "search_box_least_squares":
		search_module = import_module('.box_least_squares', __name__)
		value = getattr(search_module, name)
		globals()[name] = value
		return value
	if name in {"compute_photometric_signatures", "derive_compact_object_features"}:
		signatures_module = import_module('.signatures', __name__)
		value = getattr(signatures_module, name)
		globals()[name] = value
		return value
	main_module = import_module('.main', __name__)
	if name == 'main':
		return main_module
	try:
		value = getattr(main_module, name)
	except AttributeError as exception:
		raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from exception
	globals()[name] = value
	return value
