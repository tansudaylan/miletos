"""Box Least Squares searches for periodic transit-like signals."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from astropy.timeseries import BoxLeastSquares
from tdpy.plotting import apply_plot_theme, themed_plot

@themed_plot
def animate_box_least_squares_search(
    data_by_label: Mapping[str, np.ndarray],
    output_path,
    minimum_period_days: float,
    maximum_period_days: float,
    duration_days,
    number_of_periods: int = 120,
    maximum_points: int = 2000,
    frames_per_second: int = 12,
    target_label: str = "target",
) -> dict[str, object]:
    """Animate a BLS period scan and folded best fit for each labeled light curve.

    Each input array has columns of time [day], relative flux, and flux uncertainty.
    The returned summary contains the top-scoring trial for each light curve.
    """

    if not isinstance(data_by_label, Mapping) or not data_by_label:
        raise ValueError("data_by_label must be a nonempty mapping of labels to light curves.")
    if not 0.0 < minimum_period_days < maximum_period_days:
        raise ValueError("period bounds must be positive and increasing.")
    if number_of_periods < 2 or maximum_points < 3 or frames_per_second < 1:
        raise ValueError("number_of_periods, maximum_points, and frames_per_second are too small.")

    durations = np.atleast_1d(np.asarray(duration_days, dtype=float))
    if not np.isfinite(durations).all() or np.any(durations <= 0.0) or np.any(durations >= minimum_period_days):
        raise ValueError("duration_days must contain finite durations shorter than the minimum period.")

    periods = np.linspace(minimum_period_days, maximum_period_days, number_of_periods)
    analyses = {}
    for label, values in data_by_label.items():
        data = np.asarray(values, dtype=float)
        if data.ndim != 2 or data.shape[1] != 3 or data.shape[0] < 3:
            raise ValueError("Each light curve must have at least three rows and three columns.")
        if not np.isfinite(data).all() or np.any(data[:, 2] <= 0.0):
            raise ValueError("Light curves must contain finite data and positive uncertainties.")
        if np.any(np.diff(data[:, 0]) <= 0.0):
            raise ValueError("Light-curve times must be strictly increasing.")
        if data.shape[0] > maximum_points:
            indices = np.arange(0, data.shape[0], int(np.ceil(data.shape[0] / maximum_points)))
            data = data[indices]

        time_days, flux, uncertainty = data.T
        model = BoxLeastSquares(time_days, flux, dy=uncertainty)
        period_grid, duration_grid = np.meshgrid(periods, durations, indexing="ij")
        results = model.power(
            period_grid.ravel(), duration_grid.ravel(), objective="snr"
        )
        shape = period_grid.shape
        analyses[label] = {
            "time": time_days,
            "flux": flux,
            "uncertainty": uncertainty,
            "periods": periods,
            "model": model,
            "scores": np.asarray(results.depth_snr).reshape(shape),
            "depths": np.asarray(results.depth).reshape(shape),
            "epochs": np.asarray(results.transit_time).reshape(shape),
            "durations": np.asarray(results.duration).reshape(shape),
        }

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    labels = list(analyses)
    figure, axes = plt.subplots(
        len(labels), 2, figsize=(11.0, max(3.2, 2.7 * len(labels))), squeeze=False
    )
    period_lines = []
    candidate_marks = []
    folded_points = []
    model_lines = []
    phase_grid = np.linspace(-0.5, 0.5, 401)

    for row, label in enumerate(labels):
        period_axis, phase_axis = axes[row]
        analysis = analyses[label]
        period_axis.axhline(7.1, color="#A51C30", linestyle="--", linewidth=0.8, alpha=0.7)
        period_line, = period_axis.plot([], [], color="#176B87", linewidth=1.5, marker="o", markersize=2)
        candidate_mark, = period_axis.plot([], [], marker="o", color="#D55E00", markersize=5)
        period_axis.set_xlim(minimum_period_days, maximum_period_days)
        score_min = float(np.nanmin(analysis["scores"]))
        score_max = float(np.nanmax(analysis["scores"]))
        score_span = max(score_max - score_min, 1.0)
        period_axis.set_ylim(min(0.0, score_min) - 0.05 * score_span,
                     max(7.1, score_max) + 0.05 * score_span)
        period_axis.set_ylabel("BLS depth SNR")
        period_axis.set_title(label)
        period_lines.append(period_line)
        candidate_marks.append(candidate_mark)

        points = phase_axis.scatter([], [], s=5, color="#555555", alpha=0.45, linewidths=0)
        model_line, = phase_axis.plot([], [], color="#D55E00", linewidth=1.6)
        phase_axis.set_xlim(-0.5, 0.5)
        flux_lower, flux_upper = np.nanpercentile(analysis["flux"], [0.5, 99.5])
        padding = max(
            0.1 * (flux_upper - flux_lower),
            3.0 * float(np.median(analysis["uncertainty"])),
            1e-5,
        )
        phase_axis.set_ylim(
            min(flux_lower - padding, 1.0 - float(np.nanmax(analysis["depths"])) - padding),
            max(flux_upper + padding, 1.0 + padding),
        )
        phase_axis.set_ylabel("Relative flux")
        phase_axis.set_xlabel("Orbital phase")
        folded_points.append(points)
        model_lines.append(model_line)

    axes[-1, 0].set_xlabel("Trial period [day]")
    title = figure.suptitle(f"Miletos BLS scan | {target_label}")
    figure.text(
        0.5, 0.005,
        f"Independent searches, at most {maximum_points} points per curve; screening only",
        ha="center", fontsize=9, color="#555555",
    )
    figure.tight_layout(rect=(0.02, 0.035, 0.98, 0.96))

    def update(frame):
        frame_count = frame + 1
        title.set_text(
            f"Miletos BLS scan | {target_label} | "
            f"trial periods through {periods[frame]:.3f} day"
        )
        for row, label in enumerate(labels):
            analysis = analyses[label]
            scores = analysis["scores"]
            prefix = scores[:frame_count]
            finite = np.isfinite(prefix)
            if not finite.any():
                continue
            best_period_index, best_duration_index = np.unravel_index(
                int(np.nanargmax(prefix)), prefix.shape
            )
            period = analysis["periods"][best_period_index]
            duration = analysis["durations"][best_period_index, best_duration_index]
            epoch = analysis["epochs"][best_period_index, best_duration_index]

            period_lines[row].set_data(periods[:frame_count], np.nanmax(prefix, axis=1))
            candidate_marks[row].set_data(
                [period], [scores[best_period_index, best_duration_index]]
            )

            time_days = analysis["time"]
            phase = np.remainder(time_days - epoch + 0.5 * period, period) / period - 0.5
            folded_points[row].set_offsets(np.column_stack((phase, analysis["flux"])))
            model_time = epoch + phase_grid * period
            model_flux = analysis["model"].model(model_time, period, duration, epoch)
            model_lines[row].set_data(phase_grid, model_flux)
        return [title, *period_lines, *candidate_marks, *folded_points, *model_lines]

    animation = FuncAnimation(
        figure, update, frames=number_of_periods, interval=1000 / frames_per_second,
        blit=False, repeat=False,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    apply_plot_theme(figure)
    animation.save(output_path, writer=PillowWriter(fps=frames_per_second), dpi=90)
    plt.close(figure)

    summary = {}
    for label, analysis in analyses.items():
        index = np.unravel_index(int(np.nanargmax(analysis["scores"])), analysis["scores"].shape)
        summary[label] = {
            "period_days": float(analysis["periods"][index[0]]),
            "duration_days": float(analysis["durations"][index]),
            "depth": float(analysis["depths"][index]),
            "depth_snr": float(analysis["scores"][index]),
        }
    return {"path": output_path, "summary": summary}
def search_box_least_squares(
    time_days,
    flux,
    flux_uncertainty,
    duration_days,
    minimum_period_days,
    maximum_period_days,
    frequency_factor=1.0,
) -> dict[str, object]:
    """Search a sorted light curve and return the highest-SNR transit candidate."""

    time_days = np.asarray(time_days, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_uncertainty = np.asarray(flux_uncertainty, dtype=float)
    duration_days = np.atleast_1d(np.asarray(duration_days, dtype=float))
    if time_days.ndim != 1 or time_days.size < 3:
        raise ValueError("time_days must be a one-dimensional array with at least three points")
    if flux.shape != time_days.shape or flux_uncertainty.shape != time_days.shape:
        raise ValueError("flux and flux_uncertainty must match time_days")
    if not all(np.isfinite(values).all() for values in (time_days, flux, flux_uncertainty)):
        raise ValueError("time, flux, and uncertainty values must be finite")
    if np.any(np.diff(time_days) <= 0.0):
        raise ValueError("time_days must be strictly increasing")
    if np.any(flux_uncertainty <= 0.0):
        raise ValueError("flux_uncertainty must be positive")
    if not np.isfinite(duration_days).all() or np.any(duration_days <= 0.0):
        raise ValueError("duration_days must contain finite positive values")
    if not 0.0 < minimum_period_days < maximum_period_days:
        raise ValueError("period bounds must be positive and increasing")
    if np.any(duration_days >= minimum_period_days):
        raise ValueError("transit durations must be shorter than the minimum period")
    if not np.isfinite(frequency_factor) or frequency_factor <= 0.0:
        raise ValueError("frequency_factor must be finite and positive")

    model = BoxLeastSquares(time_days, flux, dy=flux_uncertainty)
    periodogram = model.autopower(
        duration_days,
        objective="snr",
        minimum_period=minimum_period_days,
        maximum_period=maximum_period_days,
        minimum_n_transit=3,
        frequency_factor=frequency_factor,
    )
    best_index = int(np.nanargmax(periodogram.power))
    return {
        "period_days": float(periodogram.period[best_index]),
        "transit_time_days": float(periodogram.transit_time[best_index]),
        "duration_days": float(periodogram.duration[best_index]),
        "depth": float(periodogram.depth[best_index]),
        "depth_uncertainty": float(periodogram.depth_err[best_index]),
        "depth_snr": float(periodogram.depth_snr[best_index]),
        "periodogram": periodogram,
        "model": model,
    }