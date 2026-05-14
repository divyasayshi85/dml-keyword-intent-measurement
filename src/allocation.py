"""Allocation-efficiency analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .data_generation import DatasetBundle
from .estimators import ESTIMATORS


def unit_scores(bundle: DatasetBundle, method: str, seed: int = 42) -> np.ndarray:
    """Construct unit-level scores for allocation.

    Single-ATE methods are converted to generic score-based rankings by adding
    non-domain-specific covariate variation. The optimal benchmark ranks by the
    true treatment effect.
    """
    rng = np.random.default_rng(seed)
    X, treatment, outcome = bundle.X, bundle.treatment, bundle.outcome

    if method == "Optimal":
        return bundle.true_tau.copy()

    if method == "Causal Forest":
        X_np = X.to_numpy()
        treated_model = RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=12,
            max_depth=6,
            max_features="sqrt",
            random_state=seed,
            n_jobs=1,
        )
        control_model = RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=12,
            max_depth=6,
            max_features="sqrt",
            random_state=seed + 1,
            n_jobs=1,
        )
        treated_model.fit(X_np[treatment == 1], outcome[treatment == 1])
        control_model.fit(X_np[treatment == 0], outcome[treatment == 0])
        return treated_model.predict(X_np) - control_model.predict(X_np)

    estimated_ate = ESTIMATORS[method](X, treatment, outcome, seed=seed)
    covariate_score = 0.01 * (
        0.5 * X.iloc[:, 0].to_numpy()
        - 0.3 * X.iloc[:, 1].to_numpy()
        + 0.2 * np.tanh(X.iloc[:, 2].to_numpy())
    )
    return estimated_ate + covariate_score + rng.normal(0, 0.002, len(outcome))


def allocation_efficiency(
    bundle: DatasetBundle,
    top_fraction: float = 0.25,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare allocation efficiency across methods.

    Relative efficiency is the mean true treatment effect among selected units
    divided by the optimal mean true treatment effect.
    """
    methods = ["Naive", "OLS", "IPW", "Causal Forest", "DML", "Optimal"]
    k = max(1, int(len(bundle.outcome) * top_fraction))

    scores = {method: unit_scores(bundle, method, seed) for method in methods}

    optimal_index = np.argsort(scores["Optimal"])[-k:]
    optimal_value = float(np.mean(bundle.true_tau[optimal_index]))

    rows = []
    for method in methods:
        selected_index = np.argsort(scores[method])[-k:]
        selected_value = float(np.mean(bundle.true_tau[selected_index]))
        rows.append({
            "method": method,
            "selected_true_effect": selected_value,
            "optimal_selected_true_effect": optimal_value,
            "relative_efficiency": selected_value / optimal_value,
        })

    return pd.DataFrame(rows)
