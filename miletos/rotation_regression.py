"""Explicitly synthetic period-regression benchmark transferred from Hattusa."""

import numpy as np


def run_rotation_regression_benchmark(
    number_samples=256, number_times=128, training_fraction=0.8,
    seed=42, number_trees=64,
):
    """Train a random forest on toy sinusoidal curves and predict held-out periods.

    All curves include independent white noise and ten random sinusoidal noise
    components. Periods span 1 to 10 days; signal amplitudes span 1e-3 to 1e-1,
    and noise amplitudes span 1e-4 to 1e-2. These are chosen generative inputs,
    not measured stellar variability or a survey-completeness calculation.
    """

    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as exception:
        raise ImportError("Install miletos[rotation_ml] for the rotation-regression benchmark") from exception
    if number_samples < 2 or number_times < 2 or number_trees < 1:
        raise ValueError("Provide at least two curves, two times, and one tree")
    split_index = int(number_samples * training_fraction)
    if not 0 < training_fraction < 1 or not 1 <= split_index < number_samples:
        raise ValueError("Training and held-out samples must both be nonempty")
    generator = np.random.default_rng(seed)
    time_days = np.linspace(0.0, 27.3, number_times)  # [day]
    period_days = generator.uniform(1.0, 10.0, number_samples)  # [day]
    phase_radians = generator.uniform(0.0, 2.0 * np.pi, number_samples)  # [rad]
    signal_amplitudes = 10.0 ** generator.uniform(-3.0, -1.0, number_samples)
    white_noise_amplitudes = 10.0 ** generator.uniform(-4.0, -2.0, number_samples)
    light_curves = 1.0 + signal_amplitudes[:, None] * np.sin(
        2.0 * np.pi * time_days[None, :] / period_days[:, None] + phase_radians[:, None],
    )
    light_curves += white_noise_amplitudes[:, None] * generator.normal(size=light_curves.shape)
    noise_amplitudes = 10.0 ** generator.uniform(-4.0, -2.0, (number_samples, 10))
    noise_period_days = 10.0 ** generator.uniform(-2.0, 0.4, (number_samples, 10))  # [day]
    noise_phase_radians = generator.uniform(0.0, 2.0 * np.pi, (number_samples, 10))  # [rad]
    for component in range(10):
        light_curves += noise_amplitudes[:, component, None] * np.sin(
            2.0 * np.pi * time_days[None, :] / noise_period_days[:, component, None]
            + noise_phase_radians[:, component, None],
        )
    regressor = RandomForestRegressor(random_state=seed, n_estimators=number_trees)
    regressor.fit(light_curves[:split_index], period_days[:split_index])
    predictions = regressor.predict(light_curves[split_index:])
    return {
        "time_days": time_days, "light_curves": light_curves, "period_days": period_days,
        "signal_amplitudes": signal_amplitudes, "white_noise_amplitudes": white_noise_amplitudes,
        "train_indices": np.arange(split_index), "test_indices": np.arange(split_index, number_samples),
        "predicted_period_days": predictions,
        "relative_period_errors": np.abs(predictions - period_days[split_index:]) / period_days[split_index:],
        "regressor": regressor,
    }