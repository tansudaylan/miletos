"""Shared orchestration for observational Miletos analyses."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .visualization import miletos_plot_context


PlotProduct = tuple[str, Callable[[Any, Path], Path]]


def run_observational_pipeline(
    *,
    analyzer: Callable[..., Any],
    output_path: Path,
    typefileplot: str = 'png',
    analysis_kwargs: Mapping[str, Any] | None = None,
    plot_products: Sequence[PlotProduct] = (),
) -> Any:
    """Run one observational analysis and render its products consistently."""

    if typefileplot not in {'png', 'pdf'}:
        raise ValueError("typefileplot must be 'png' or 'pdf'")
    keyword_arguments = {} if analysis_kwargs is None else dict(analysis_kwargs)
    with miletos_plot_context():
        result = analyzer(**keyword_arguments)
        for filename, plot_product in plot_products:
            plot_product(result, output_path / f'{filename}.{typefileplot}')
    return result