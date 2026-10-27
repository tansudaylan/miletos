"""Native time-series storage and operations for Miletos photometry."""

from dataclasses import dataclass, field

import numpy as np
from astropy import units
from astropy.stats import sigma_clip
from astropy.time import Time, TimeDelta
from astropy.timeseries import TimeSeries, aggregate_downsample


def time_values_days(time):
    """Return the existing day coordinates, retaining an absolute Time's format."""
    if isinstance(time, Time):
        return np.asarray(time.value, dtype=float)
    if isinstance(time, TimeDelta):
        return time.to_value(units.day)
    return units.Quantity(time, units.day).to_value(units.day)


@dataclass
class LightCurve:
    """Measured flux and errors with explicit day coordinates or Astropy times.

    Numeric times are relative day coordinates. Their origin is not inferred.
    Missing uncertainties remain NaN rather than being estimated from scatter.
    """

    time: object
    flux: object
    flux_err: object = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.time, (Time, TimeDelta)):
            self.time = units.Quantity(self.time, units.day)
        self.flux = units.Quantity(self.flux)
        if self.flux_err is None:
            self.flux_err = np.full(self.flux.shape, np.nan) * self.flux.unit
        else:
            self.flux_err = units.Quantity(self.flux_err, self.flux.unit)
        if (self.flux.ndim != 1 or self.time.shape != self.flux.shape
                or self.flux_err.shape != self.flux.shape):
            raise ValueError('Time, flux, and flux_err must be aligned one-dimensional arrays')

    @property
    def colnames(self):
        return ['time', 'flux', 'flux_err']

    def __len__(self):
        return len(self.time)

    def __getitem__(self, selection):
        if isinstance(selection, str):
            if selection not in self.colnames:
                raise KeyError(selection)
            return getattr(self, selection)
        return type(self)(self.time[selection], self.flux[selection],
                          self.flux_err[selection], self.meta.copy())

    def copy(self):
        return type(self)(self.time.copy(), self.flux.copy(), self.flux_err.copy(), self.meta.copy())

    def remove_nans(self):
        return self[~np.isnan(self.flux)]

    def normalize(self):
        median = np.nanmedian(self.flux)
        if not np.isfinite(median.value) or median.value == 0:
            raise ValueError('Flux normalization requires a finite nonzero median')
        return type(self)(self.time.copy(), self.flux / median, self.flux_err / median, self.meta.copy())

    def remove_outliers(self, sigma=5):
        clipped = sigma_clip(self.flux.value, sigma=sigma)
        return self[~np.ma.getmaskarray(clipped)]

    def fold(self, period, epoch_time=0):
        period_days = units.Quantity(period, units.day).to_value(units.day)
        if not np.isfinite(period_days) or period_days <= 0:
            raise ValueError('Fold period must be finite and positive')
        if isinstance(self.time, Time) and isinstance(epoch_time, Time):
            offsets = (self.time - epoch_time).to_value(units.day)
        else:
            epoch = (float(epoch_time.value) if isinstance(epoch_time, Time)
                     else units.Quantity(epoch_time, units.day).to_value(units.day))
            offsets = time_values_days(self.time) - epoch
        phase = (offsets + 0.5 * period_days) % period_days - 0.5 * period_days
        order = np.argsort(phase, kind='stable')
        metadata = dict(self.meta, period_days=float(period_days))
        return type(self)(TimeDelta(phase[order], format='jd'), self.flux[order],
                          self.flux_err[order], metadata)

    def bin(self, time_bin_size=0.001):
        """Average bins and propagate independent errors as sqrt(sum(error**2))/N."""
        width = units.Quantity(time_bin_size, units.day)
        if not np.isfinite(width.value) or width <= 0 * units.day:
            raise ValueError('Bin width must be finite and positive')
        coordinates = time_values_days(self.time)
        series_time = self.time.copy() if isinstance(self.time, Time) else Time(coordinates, format='jd', scale='tdb')
        series = TimeSeries(time=series_time, data={
            'flux': self.flux.value, 'variance': self.flux_err.value**2,
            'count': np.ones(len(self)), 'errors': np.isfinite(self.flux_err).astype(int),
        })
        binned = aggregate_downsample(series, time_bin_size=width, aggregate_func=np.nansum)
        counts = np.asarray(binned['count'].filled(0), dtype=float)
        keep = counts > 0
        means = np.asarray(binned['flux'].filled(np.nan))[keep] / counts[keep]
        errors = np.sqrt(np.asarray(binned['variance'].filled(np.nan))[keep]) / counts[keep]
        complete = np.asarray(binned['errors'].filled(0))[keep] == counts[keep]
        errors[~complete] = np.nan
        centers = (binned.time_bin_start + binned.time_bin_size / 2).jd[keep]
        if isinstance(self.time, Time):
            time = Time(centers, format='jd', scale=self.time.scale)
            time.format = self.time.format
        else:
            time = TimeDelta(centers, format='jd') if isinstance(self.time, TimeDelta) else centers * units.day
        return type(self)(time, means * self.flux.unit, errors * self.flux.unit, self.meta.copy())

    def scatter(self, **kwargs):
        import matplotlib.pyplot as plt

        coordinates = time_values_days(self.time)
        if 'period_days' in self.meta:
            coordinates = coordinates / self.meta['period_days']
        return plt.scatter(coordinates, self.flux.value, **kwargs)

    def plot(self, **kwargs):
        return self.scatter(**kwargs)


def as_light_curve(curve):
    """Copy array columns into the native representation without external methods."""
    if isinstance(curve, LightCurve):
        return curve.copy()
    names = getattr(curve, 'colnames', curve.keys() if isinstance(curve, dict) else ())
    error = curve['flux_err'] if 'flux_err' in names else None
    return LightCurve(curve['time'], curve['flux'], error, dict(getattr(curve, 'meta', {})))