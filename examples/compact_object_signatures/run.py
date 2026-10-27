#!/usr/bin/env python3
"""Plot compact-object photometric amplitudes computed by Ephesos through Miletos."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tdpy.cli import add_plot_arguments
from tdpy.plotting import save_plot, themed_plot
from tdpy.verbosity import print

from miletos.paths import get_example_visuals_path
from miletos.signatures import compute_photometric_signatures


PANEL_TITLES = {
    "beaming": "Doppler beaming",
    "ellipsoidal": "Ellipsoidal variation",
    "self_lensing": "Self-lensing",
}
COLORS = ("#1B6CA8", "#D18B00", "#2E7D32")


@themed_plot
def run_bokeh_example(
    period_days: np.ndarray,
    companion_mass_solar: np.ndarray,
    signatures: dict[str, np.ndarray],
    output_path: Path,
) -> Path:
    """Write interactive period-amplitude panels for the three Ephesos models."""
    from bokeh.layouts import row
    from bokeh.models import HoverTool
    from bokeh.plotting import figure
    from tdpy.plotting import save_bokeh_html

    plots = []
    for name, title in PANEL_TITLES.items():
        plot = figure(
            title=title,
            x_axis_type="log",
            y_axis_type="log",
            x_range=(float(period_days.min()), float(period_days.max())),
            y_range=(5e-3, 1e3),
            width=420,
            height=360,
            sizing_mode="stretch_width",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        for mass_index, mass_solar in enumerate(companion_mass_solar.ravel()):
            plot.line(
                period_days.ravel(),
                signatures[name][:, mass_index],
                color=COLORS[mass_index],
                line_width=2.2,
                legend_label=f"{mass_solar:g} solar masses",
            )
        plot.xaxis.axis_label = "Orbital period [day]"
        plot.yaxis.axis_label = "Predicted amplitude [ppt]"
        plot.grid.visible = False
        plot.legend.location = "top_right"
        plot.add_tools(HoverTool(tooltips=[("Period [day]", "$x"), ("Amplitude [ppt]", "$y")]))
        plots.append(plot)
    return Path(save_bokeh_html(row(*plots, sizing_mode="stretch_width"), output_path,
                                title="Compact-object photometric signatures"))


@themed_plot
def run_example(
    output_path: Path | None = None,
    bokeh_output_path: Path | None = None,
    typeplotback: str = "white",
) -> dict[str, np.ndarray]:
    """Calculate amplitudes through Miletos and plot the Ephesos predictions."""

    period_days = np.geomspace(0.3, 30.0, 200)[:, None]  # [day]
    companion_mass_solar = np.array([5.0, 30.0, 180.0])[None, :]  # [M_Sun]
    signatures = compute_photometric_signatures(
        period_days,
        companion_mass_solar,
        stellar_radius_solar=1.0,
        stellar_mass_solar=1.0,
        stellar_density_cgs=1.41,
    )

    figure, axes = plt.subplots(
        1, 3, figsize=(12.5, 4.0), sharex=True, sharey=True, constrained_layout=True
    )
    for axis, (name, title) in zip(axes, PANEL_TITLES.items()):
        for mass_index, mass_solar in enumerate(companion_mass_solar.ravel()):
            axis.plot(
                period_days.ravel(),
                signatures[name][:, mass_index],
                color=COLORS[mass_index],
                linewidth=2.2,
                label=rf"{mass_solar:g} $M_\odot$",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(period_days.min(), period_days.max())
        axis.set_ylim(5e-3, 1e3)
        axis.set_xlabel("Orbital period [day]")
        axis.set_title(title)
        axis.grid(False)

    axes[0].set_ylabel("Predicted amplitude [ppt]")
    legend = axes[-1].legend(
        title="Companion mass", frameon=True, fancybox=True, framealpha=1.0
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    figure.suptitle(
        r"Compact-object signals for a $1\,M_\odot$, $1\,R_\odot$ source",
        fontweight="bold",
    )

    output_path = Path(
        output_path
        or get_example_visuals_path("compact_object_signatures")
        / "compact_object_signatures.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    save_plot(figure, output_path, dpi=300 if output_path.suffix == ".png" else None,
              bbox_inches="tight")
    plt.close(figure)
    if bokeh_output_path is not None:
        run_bokeh_example(
            period_days, companion_mass_solar, signatures, Path(bokeh_output_path),
            typeplotback=typeplotback,
        )
    return signatures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_arguments(parser)
    parser.add_argument("--typeplotback", choices=("white", "dark"), default="white")
    parser.add_argument("--bokeh", action="store_true", help="also write interactive HTML")
    parser.add_argument(
        "--output-directory", type=Path,
        default=get_example_visuals_path("compact_object_signatures"),
    )
    arguments = parser.parse_args(argv)
    output_path = arguments.output_directory / f"compact_object_signatures.{arguments.typefileplot}"
    html_path = arguments.output_directory / "compact_object_signatures.html" if arguments.bokeh else None
    run_example(
        output_path,
        bokeh_output_path=html_path,
        typeplotback=arguments.typeplotback,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
