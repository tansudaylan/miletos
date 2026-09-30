"""Reproduce the Daylan et al. (2021) TOI-1233 transit results."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from .paths import get_data_path, get_repository_path
from .pipeline import run_observational_pipeline


PAPER_DOI = '10.3847/1538-3881/abd73e'


@dataclass(frozen=True)
class PublishedTransit:
    """A posterior summary reported in Tables 11--14 of Daylan et al. (2021)."""

    planet: str
    epoch_bjd: float
    epoch_uncertainty_days: float
    period_days: float
    period_uncertainty_days: float
    depth: float
    depth_uncertainty: float
    duration_days: float
    duration_uncertainty_days: float


@dataclass(frozen=True)
class RecoveredTransit:
    """A weighted trapezoid fit to the publication-era TESS observations."""

    planet: str
    epoch_bjd: float
    epoch_uncertainty_days: float
    period_days: float
    period_uncertainty_days: float
    depth: float
    depth_uncertainty: float
    duration_days: float
    duration_uncertainty_days: float
    ingress_fraction: float


@dataclass(frozen=True)
class Daylan2021Reproduction:
    """Observations and recovered transit parameters for HD 108236 b--e."""

    time_bjd: np.ndarray
    raw_flux: np.ndarray
    flux_uncertainty: np.ndarray
    baseline_flux: np.ndarray
    detrended_flux: np.ndarray
    published: tuple[PublishedTransit, ...]
    recovered: tuple[RecoveredTransit, ...]
    maximum_sigma_difference: float

    @property
    def reproduces_published_results(self) -> bool:
        """Whether every fitted summary agrees with the paper within three sigma."""

        return self.maximum_sigma_difference <= 3.0


PUBLISHED_TRANSITS = (
    PublishedTransit('b', 2458572.1128, 0.00335, 3.79523, 0.000455, 0.000302, 0.000031, 2.30 / 24.0, 0.135 / 24.0),
    PublishedTransit('c', 2458572.3949, 0.00225, 6.20370, 0.000580, 0.000517, 0.000038, 2.913 / 24.0, 0.095 / 24.0),
    PublishedTransit('d', 2458571.3368, 0.00140, 14.17555, 0.001045, 0.000889, 0.000053, 3.734 / 24.0, 0.0575 / 24.0),
    PublishedTransit('e', 2458586.5677, 0.00140, 19.5917, 0.00210, 0.001175, 0.000069, 4.013 / 24.0, 0.0685 / 24.0),
)


def phase_time(time_bjd: np.ndarray, epoch_bjd: float, period_days: float) -> np.ndarray:
    """Return signed time from the nearest transit center in days."""

    return (time_bjd - epoch_bjd + 0.5 * period_days) % period_days - 0.5 * period_days


def trapezoid_transit(
    time_bjd: np.ndarray,
    epoch_bjd: float,
    period_days: float,
    depth: float,
    duration_days: float,
    ingress_fraction: float,
) -> np.ndarray:
    """Evaluate a unit-baseline periodic trapezoid transit model."""

    absolute_phase = np.abs(phase_time(time_bjd, epoch_bjd, period_days))
    half_duration = 0.5 * duration_days
    flat_half_duration = half_duration * (1.0 - ingress_fraction)
    occulted_fraction = np.clip(
        (half_duration - absolute_phase) / (half_duration - flat_half_duration),
        0.0,
        1.0,
    )
    return 1.0 - depth * occulted_fraction


def _load_publication_photometry(data_path: Path) -> tuple[np.ndarray, ...]:
    observations = []
    for sector in (10, 11):
        path = data_path / f'TESS_PDCSAP_FLUX_Sector{sector}.csv'
        if not path.exists():
            raise FileNotFoundError(
                f'Missing publication-era TESS photometry: {path}. '
                'Both Sector 10 and Sector 11 are required.'
            )
        observations.append(np.loadtxt(path, delimiter=','))
    time_bjd, flux, flux_uncertainty = np.vstack(observations).T
    order = np.argsort(time_bjd)
    return time_bjd[order], flux[order], flux_uncertainty[order]


def _detrend_photometry(
    time_bjd: np.ndarray,
    flux: np.ndarray,
    published: tuple[PublishedTransit, ...],
) -> tuple[np.ndarray, np.ndarray]:
    transit_mask = np.logical_or.reduce(
        [
            np.abs(phase_time(time_bjd, transit.epoch_bjd, transit.period_days))
            < 0.8 * transit.duration_days
            for transit in published
        ]
    )
    baseline_input = flux.copy()
    baseline_input[transit_mask] = np.interp(
        time_bjd[transit_mask],
        time_bjd[~transit_mask],
        flux[~transit_mask],
    )
    cadence_days = np.median(np.diff(time_bjd))
    baseline = gaussian_filter1d(
        baseline_input,
        0.4 / cadence_days,
        mode='nearest',
    )
    return baseline, flux / baseline


def _fit_transit(
    time_bjd: np.ndarray,
    detrended_flux: np.ndarray,
    flux_uncertainty: np.ndarray,
    transit: PublishedTransit,
    all_transits: tuple[PublishedTransit, ...],
) -> RecoveredTransit:
    selected = np.abs(
        phase_time(time_bjd, transit.epoch_bjd, transit.period_days)
    ) < 0.35
    for other in all_transits:
        if other.planet != transit.planet:
            selected &= np.abs(
                phase_time(time_bjd, other.epoch_bjd, other.period_days)
            ) > 0.6 * other.duration_days

    fit_time = time_bjd[selected]
    fit_flux = detrended_flux[selected]
    fit_uncertainty = flux_uncertainty[selected]

    def residual(parameters: np.ndarray) -> np.ndarray:
        model = trapezoid_transit(
            fit_time,
            transit.epoch_bjd + parameters[0],
            transit.period_days + parameters[1],
            parameters[2],
            parameters[3],
            parameters[4],
        )
        return (fit_flux - model) / fit_uncertainty

    solution = least_squares(
        residual,
        [0.0, 0.0, transit.depth, transit.duration_days, 0.15],
        bounds=(
            [-0.05, -0.02, 0.0, 0.5 * transit.duration_days, 0.02],
            [0.05, 0.02, 0.003, 1.5 * transit.duration_days, 0.45],
        ),
        diff_step=1e-5,
        x_scale=[0.01, 0.005, 0.0005, 0.03, 0.1],
    )
    degrees_of_freedom = fit_time.size - solution.x.size
    covariance = np.linalg.pinv(solution.jac.T @ solution.jac)
    covariance *= np.sum(solution.fun**2) / degrees_of_freedom
    uncertainty = np.sqrt(np.diag(covariance))
    return RecoveredTransit(
        transit.planet,
        transit.epoch_bjd + solution.x[0],
        uncertainty[0],
        transit.period_days + solution.x[1],
        uncertainty[1],
        solution.x[2],
        uncertainty[2],
        solution.x[3],
        uncertainty[3],
        solution.x[4],
    )


def _sigma_differences(
    published: tuple[PublishedTransit, ...],
    recovered: tuple[RecoveredTransit, ...],
) -> list[float]:
    differences = []
    for reference, estimate in zip(published, recovered, strict=True):
        for value_name, uncertainty_name in (
            ('epoch_bjd', 'epoch_uncertainty_days'),
            ('period_days', 'period_uncertainty_days'),
            ('depth', 'depth_uncertainty'),
            ('duration_days', 'duration_uncertainty_days'),
        ):
            combined_uncertainty = np.hypot(
                getattr(reference, uncertainty_name),
                getattr(estimate, uncertainty_name),
            )
            differences.append(
                abs(getattr(estimate, value_name) - getattr(reference, value_name))
                / combined_uncertainty
            )
    return differences


def reproduce_daylan2021(data_path: Path | None = None) -> Daylan2021Reproduction:
    """Fit the four published transits in the Sector 10 and 11 PDC data."""

    if data_path is None:
        data_path = get_data_path() / 'TOI-1233' / 'data'
    time_bjd, raw_flux, raw_uncertainty = _load_publication_photometry(data_path)
    baseline, detrended_flux = _detrend_photometry(
        time_bjd,
        raw_flux,
        PUBLISHED_TRANSITS,
    )
    detrended_uncertainty = raw_uncertainty / baseline
    recovered = tuple(
        _fit_transit(
            time_bjd,
            detrended_flux,
            detrended_uncertainty,
            transit,
            PUBLISHED_TRANSITS,
        )
        for transit in PUBLISHED_TRANSITS
    )
    sigma_differences = _sigma_differences(PUBLISHED_TRANSITS, recovered)
    return Daylan2021Reproduction(
        time_bjd,
        raw_flux,
        detrended_uncertainty,
        baseline,
        detrended_flux,
        PUBLISHED_TRANSITS,
        recovered,
        max(sigma_differences),
    )


def run_daylan2021_reproduction(typefileplot: str = 'png') -> Daylan2021Reproduction:
    """Run the publication reproduction and write its figures and comparison table."""

    from .visualization import (
        plot_daylan2021_light_curve,
        plot_daylan2021_phase_curves,
    )

    output_path = get_repository_path() / 'examples' / 'TOI-1233' / 'Daylan2021'
    result = run_observational_pipeline(
        analyzer=reproduce_daylan2021,
        output_path=output_path,
        typefileplot=typefileplot,
        plot_products=(
            ('light_curve', plot_daylan2021_light_curve),
            ('phase_curves', plot_daylan2021_phase_curves),
        ),
    )
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_path = output_path / 'parameter_comparison.csv'
    with comparison_path.open('w', encoding='utf-8') as file:
        file.write(
            'planet,quantity,published,published_uncertainty,recovered,'
            'recovered_uncertainty,sigma_difference\n'
        )
        for reference, estimate in zip(result.published, result.recovered, strict=True):
            for quantity, uncertainty in (
                ('epoch_bjd', 'epoch_uncertainty_days'),
                ('period_days', 'period_uncertainty_days'),
                ('depth', 'depth_uncertainty'),
                ('duration_days', 'duration_uncertainty_days'),
            ):
                combined = np.hypot(
                    getattr(reference, uncertainty),
                    getattr(estimate, uncertainty),
                )
                sigma = abs(
                    getattr(estimate, quantity) - getattr(reference, quantity)
                ) / combined
                file.write(
                    f'{reference.planet},{quantity},'
                    f'{getattr(reference, quantity):.10g},'
                    f'{getattr(reference, uncertainty):.10g},'
                    f'{getattr(estimate, quantity):.10g},'
                    f'{getattr(estimate, uncertainty):.10g},{sigma:.6f}\n'
                )
    return result