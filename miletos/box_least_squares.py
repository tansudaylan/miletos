"""Box Least Squares searches for periodic transit-like signals."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from astropy.timeseries import BoxLeastSquares
from matplotlib.text import Text
from tdpy.plotting import apply_plot_theme, plot_background_colors, save_figure, themed_plot

FIGURE_FONT_SIZE_PT = 12.0  # [pt]


def _set_figure_font_size(figure):
    for text in figure.findobj(match=Text):
        text.set_fontsize(FIGURE_FONT_SIZE_PT)


def fixed_grid_box_least_squares(light_curve):
    """Search Albus's fixed grid with Astropy, without a Lightkurve dependency.

    Accept a time-series object with time, flux, and optional flux_err columns.
    Retain the likelihood objective and ignore all errors if any are missing,
    matching the historical Albus fit. The result is an Astropy BLS result
    with best-fit attributes supplied for existing Albus measurement callers.
    """
    from astropy import units

    minimum_period_days = 1  # [day]
    maximum_period_days = 10  # [day]
    periods = np.logspace(np.log10(minimum_period_days), np.log10(maximum_period_days), 2048)
    durations = periods * 0.01
    time, flux = light_curve['time'], light_curve['flux']
    keep = ~np.isnan(flux)
    time, flux = time[keep], flux[keep]
    errors = light_curve['flux_err'][keep] if 'flux_err' in light_curve.colnames else None
    if errors is not None and not np.isfinite(errors).all():
        errors = None
    model = BoxLeastSquares(time, flux, errors)
    result = model.power(periods, durations)
    for name in ('period', 'duration'):
        if not isinstance(result[name], units.Quantity):
            result[name] = result[name] * units.day
    if not isinstance(result.power, units.Quantity):
        result.power = result.power * units.dimensionless_unscaled
    index = int(np.argmax(result.power))
    for name in ('period', 'duration', 'transit_time', 'depth'):
        result[f'{name}_at_max_power'] = result[name][index]
    result['snr'] = result.depth_snr
    result['bls_obj'] = model
    return result


lightkurve_box_least_squares = fixed_grid_box_least_squares


def select_lightkurve_bls_candidates(results):
    """Rank up to four distinct periods using Albus's original peak selection.

    The raw-power maximum is first, followed by the median-detrended ranking
    outside the inclusive 0.8--1.2 interval around each selected period. The
    first returned power remains raw and subsequent powers remain detrended.
    Candidate indices align the fitted epochs and durations with each period.
    """
    from scipy.signal import medfilt

    periods = np.array(results.period.value)
    powers = np.array(results.power.value)
    if periods.ndim != 1 or powers.ndim != 1 or periods.shape != powers.shape or periods.size == 0:
        raise ValueError('BLS periods and powers must be nonempty aligned one-dimensional arrays')
    if not np.isfinite(periods).all() or np.any(periods <= 0) or not np.isfinite(powers).all():
        raise ValueError('BLS periods must be finite and positive, and powers must be finite')
    index = np.argmax(results.power)
    best_period = results.period[index]
    detrended_power = powers - medfilt(powers, kernel_size=101)
    high_powers = [results.power[index].value]
    high_periods = [best_period.value]
    candidate_indices = [index]
    for candidate_index in np.argsort(detrended_power)[::-1]:
        period = periods[candidate_index]
        if all(not 0.8 * selected <= period <= 1.2 * selected for selected in high_periods):
            high_periods.append(period)
            high_powers.append(detrended_power[candidate_index])
            candidate_indices.append(candidate_index)
        if len(high_periods) >= 4:
            break
    return {
        'periods': high_periods,
        'powers': high_powers,
        'best_index': index,
        'candidate_indices': candidate_indices,
        'detrended_power': detrended_power,
    }


def lightkurve_bls_results(results, folder='', ID='', plot='', candidate_ephemerides=False,
                           typeplotback='white', typefileplot='png'):
    """Return Albus-compatible BLS candidates and optionally render their powers."""
    selected = select_lightkurve_bls_candidates(results)
    periods = np.array(results.period.value)
    powers = selected['detrended_power']
    index = selected['best_index']
    best_period = results.period[index]
    t0 = results.transit_time[index]
    duration = results.duration[index]
    if plot != '':
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(6.5, 3.8))
        num_bins = 48
        bins = np.logspace(np.log10(1), np.log10(10), num_bins + 1)  # [day]
        bin_centers, bin_means, bin_stds = [], [], []
        for bin_index in range(num_bins):
            selection = (periods >= bins[bin_index]) & (periods < bins[bin_index + 1])
            bin_periods, bin_powers = periods[selection], powers[selection]
            if bin_powers.size == 0:
                continue
            lower_quartile, upper_quartile = np.percentile(bin_powers, [25, 75])
            iqr = upper_quartile - lower_quartile
            filtered = bin_powers[(bin_powers >= lower_quartile - 1.5 * iqr)
                                  & (bin_powers <= upper_quartile + 1.5 * iqr)]
            bin_centers.append(np.median(bin_periods))
            bin_means.append(np.median(filtered))
            bin_stds.append(np.std(filtered))
        bin_centers = np.array(bin_centers)
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)
        background, data_color = plot_background_colors(typeplotback)
        is_dark = background == 'black'
        median_color = '#FFB000' if is_dark else '#D55E00'
        alias_color = '#FF7A72' if is_dark else '#A51417'
        spread_color = 'white' if is_dark else 'gray'
        axis.semilogx(periods, powers, color=data_color, linestyle='None', marker='.',
                      markersize=2.5, label=f'BLS power for {ID}')
        axis.semilogx(bin_centers, bin_means, color=median_color, linestyle='None',
                      marker='o', markersize=4, label='Binned median')
        axis.scatter(best_period.value, powers[index], marker='D', s=36,
                 color=median_color, edgecolor=data_color, zorder=4,
                 label='Selected period')
        alias_label_used = False
        for harmonic in range(1, 6):
            alias_label = 'Integer aliases' if not alias_label_used else '_nolegend_'
            if best_period.value / harmonic > 1:
                axis.axvline(best_period.value / harmonic, ls=':', color=alias_color, alpha=0.75,
                             label=alias_label)
                alias_label_used = True
            if best_period.value * harmonic < 10:
                axis.axvline(best_period.value * harmonic, ls=':', color=alias_color, alpha=0.75,
                             label=alias_label if not alias_label_used else '_nolegend_')
                alias_label_used = True
        for multiple, alpha in ((3, 0.08), (2, 0.12), (1, 0.18)):
            axis.fill_between(bin_centers, bin_means - multiple * bin_stds,
                              bin_means + multiple * bin_stds, color=spread_color,
                              alpha=max(alpha, 0.16) if is_dark else alpha,
                              label=f'{multiple} sigma spread')
        axis.set_xlabel('Period [day]')
        axis.set_ylabel('BLS power [1]')
        figure.set_size_inches(6.5, 3.8)
        figure.suptitle(f'BLS periodogram for {ID}', y=0.99)
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.91), ncol=3)
        figure.set_layout_engine('tight', rect=(0., 0., 1., 0.74))
        _set_figure_font_size(figure)
        figure = apply_plot_theme(figure, typeplotback)
        if plot == 'save':
            filename = Path(folder) / f'{ID}_blsplot'
            save_figure(figure, filename, typefileplot, typeplotback,
                        close_figure=True, font_size=FIGURE_FONT_SIZE_PT)
        if plot == 'show':
            plt.show()
        plt.close('all')
    if candidate_ephemerides:
        t0 = results.transit_time[selected['candidate_indices']]
        duration = results.duration[selected['candidate_indices']]
    return selected['periods'], selected['powers'], best_period, t0, duration


def fold_lightkurve(flatlc, best_period, t0, plot='', ID='', folder='', bin=False,
                   time_bin_size=0.001, output=False, typeplotback='white', typefileplot='png'):
    """Fold and optionally bin native photometry, retaining Albus's return convention.

    ``time_bin_size`` is in days for a time-based folded curve. Binned calls
    always return the binned curve. Unbinned calls return it only with output.
    """
    import matplotlib.pyplot as plt
    from .light_curve import as_light_curve

    folded_lc = as_light_curve(flatlc).fold(period=best_period, epoch_time=t0)
    try:
        period = best_period.value
    except BaseException:
        period = best_period
    if bin:
        binned_lc = folded_lc.bin(time_bin_size=time_bin_size)
    if plot != '':
        if bin:
            data_color = plot_background_colors(typeplotback)[1]
            binned_lc.scatter(label='Binned data', color=data_color)
        else:
            data_color = plot_background_colors(typeplotback)[1]
            folded_lc.scatter(label='Observed flux', color=data_color)
        axis = plt.gca()
        axis.set_xlabel('Time from transit [day]')
        axis.set_ylabel('Normalized flux [1]')
        rounded_period = str(round(period, 3))
        figure = plt.gcf()
        figure.set_size_inches(6.5, 3.8)
        handles, labels = axis.get_legend_handles_labels()
        figure.suptitle(f'ID {ID} folded light curve at period {rounded_period} days', y=0.99)
        figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.91), ncol=1)
        if axis.get_legend() is not None:
            axis.get_legend().remove()
        figure.set_layout_engine('tight', rect=(0., 0., 1., 0.80))
        _set_figure_font_size(figure)
        apply_plot_theme(figure, typeplotback)
        if plot == 'save':
            filename = Path(folder) / f'{ID}_{rounded_period}_foldedlc.{typefileplot}'
            save_figure(figure, filename, typefileplot, typeplotback,
                        close_figure=True, font_size=FIGURE_FONT_SIZE_PT)
        if plot == 'show':
            plt.show()
    if plot != '' or not bin:
        plt.close('all')
    if bin:
        return binned_lc
    if output:
        return folded_lc
    return None


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
        len(labels), 2, figsize=(11.0, max(3.2, 2.7 * len(labels))), squeeze=False,
        sharex='col',
    )
    period_lines = []
    candidate_marks = []
    folded_points = []
    model_lines = []
    data_color = plot_background_colors()[1]
    phase_grid = np.linspace(-0.5, 0.5, 401)

    for row, label in enumerate(labels):
        period_axis, phase_axis = axes[row]
        analysis = analyses[label]
        background, data_color = plot_background_colors()
        is_dark = background == 'black'
        period_axis.axhline(7.1, color="#FF7A72" if is_dark else "#A51C30",
                    linestyle="--", linewidth=0.8, alpha=0.8,
                    label="Screening threshold 7.1")
        period_line, = period_axis.plot([], [], color=data_color, linestyle="None", marker=".",
                        markersize=3, label="Trial-period depth SNR")
        candidate_mark, = period_axis.plot([], [], linestyle="None", marker="o",
                           color="#FFB000" if is_dark else "#D55E00",
                           markersize=5, label="Best candidate")
        period_axis.set_xlim(minimum_period_days, maximum_period_days)
        score_min = float(np.nanmin(analysis["scores"]))
        score_max = float(np.nanmax(analysis["scores"]))
        score_span = max(score_max - score_min, 1.0)
        period_axis.set_ylim(min(0.0, score_min) - 0.05 * score_span,
                     max(7.1, score_max) + 0.05 * score_span)
        period_axis.set_ylabel("BLS depth SNR [1]")
        period_axis.set_title(label)
        period_lines.append(period_line)
        candidate_marks.append(candidate_mark)

        points = phase_axis.scatter([], [], s=5, color=data_color, alpha=0.45, linewidths=0,
                        label="Folded data")
        model_line, = phase_axis.plot([], [], color="#FFB000" if is_dark else "#D55E00",
                          linewidth=1.6,
                          label="Box model")
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
        phase_axis.set_ylabel("Relative flux [1]")
        folded_points.append(points)
        model_lines.append(model_line)

    axes[-1, 0].set_xlabel("Trial period [day]")
    axes[-1, 1].set_xlabel("Orbital phase [1]")
    title = figure.suptitle(f"Miletos BLS scan | {target_label}", y=0.99)
    period_handles, period_labels = axes[0, 0].get_legend_handles_labels()
    phase_handles, phase_labels = axes[0, 1].get_legend_handles_labels()
    figure.legend(period_handles + phase_handles, period_labels + phase_labels,
                  loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=3)
    figure.text(
        0.5, 0.015,
        f"Independent searches, at most {maximum_points} points per curve; screening only",
        ha="center", fontsize=FIGURE_FONT_SIZE_PT, color="black",
    )
    figure.tight_layout(rect=(0.02, 0.09, 0.98, 0.80))

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
    _set_figure_font_size(figure)
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