"""Treatment effect estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _validate_inputs(treatment: np.ndarray, outcome: np.ndarray) -> None:
    """Validate treatment/outcome arrays."""
    if treatment.ndim != 1 or outcome.ndim != 1:
        raise ValueError("treatment and outcome must be one-dimensional arrays.")
    if len(treatment) != len(outcome):
        raise ValueError("treatment and outcome must have the same length.")
    if np.sum(treatment == 1) == 0 or np.sum(treatment == 0) == 0:
        raise ValueError("Both treated and control observations are required.")


def naive_difference_in_means(
    X: pd.DataFrame,
    treatment: np.ndarray,
    outcome: np.ndarray,
    seed: int = 42,
) -> float:
    """Difference in average outcomes between treated and control groups."""
    del X, seed
    _validate_inputs(treatment, outcome)
    return float(np.mean(outcome[treatment == 1]) - np.mean(outcome[treatment == 0]))


def ols_with_controls(
    X: pd.DataFrame,
    treatment: np.ndarray,
    outcome: np.ndarray,
    seed: int = 42,
) -> float:
    """OLS treatment coefficient controlling for observed covariates."""
    del seed
    _validate_inputs(treatment, outcome)
    design = np.column_stack([treatment, X.to_numpy()])
    model = LinearRegression().fit(design, outcome)
    return float(model.coef_[0])


def propensity_scores(X: pd.DataFrame, treatment: np.ndarray) -> np.ndarray:
    """Estimate propensity scores with regularized logistic regression."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="lbfgs"),
    )
    model.fit(X, treatment)
    return np.clip(model.predict_proba(X)[:, 1], 0.02, 0.98)


def ipw_estimator(
    X: pd.DataFrame,
    treatment: np.ndarray,
    outcome: np.ndarray,
    seed: int = 42,
) -> float:
    """Inverse probability weighted ATE estimator."""
    del seed
    _validate_inputs(treatment, outcome)
    ps = propensity_scores(X, treatment)
    return float(np.mean(treatment * outcome / ps - (1 - treatment) * outcome / (1 - ps)))


def dml_plr_estimator(
    X: pd.DataFrame,
    treatment: np.ndarray,
    outcome: np.ndarray,
    seed: int = 42,
    n_splits: int = 5,
) -> float:
    """Partially linear Double Machine Learning estimator.

    Nuisance models are estimated out of fold and treatment effect is estimated
    by regressing residualized outcome on residualized treatment.
    """
    _validate_inputs(treatment, outcome)

    X_np = X.to_numpy()
    y_resid = np.zeros_like(outcome, dtype=float)
    t_resid = np.zeros_like(treatment, dtype=float)

    outcome_model = ExtraTreesRegressor(
        n_estimators=100,
        min_samples_leaf=12,
        max_depth=6,
        max_features="sqrt",
        random_state=seed,
        n_jobs=1,
    )
    treatment_model = ExtraTreesClassifier(
        n_estimators=100,
        min_samples_leaf=12,
        max_depth=6,
        max_features="sqrt",
        random_state=seed + 1,
        n_jobs=1,
    )

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(X_np, treatment):
        y_model = clone(outcome_model)
        t_model = clone(treatment_model)

        y_model.fit(X_np[train_idx], outcome[train_idx])
        t_model.fit(X_np[train_idx], treatment[train_idx])

        y_resid[test_idx] = outcome[test_idx] - y_model.predict(X_np[test_idx])
        t_resid[test_idx] = treatment[test_idx] - t_model.predict_proba(X_np[test_idx])[:, 1]

    denominator = float(np.dot(t_resid, t_resid))
    if denominator <= 1e-12:
        raise ValueError("DML residualized treatment has near-zero variance.")

    return float(np.dot(t_resid, y_resid) / denominator)


def causal_forest_t_learner(
    X: pd.DataFrame,
    treatment: np.ndarray,
    outcome: np.ndarray,
    seed: int = 42,
) -> float:
    """Causal forest-style heterogeneous treatment effect benchmark.

    This public benchmark uses a random-forest T-learner.
    """
    _validate_inputs(treatment, outcome)
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

    tau_hat = treated_model.predict(X_np) - control_model.predict(X_np)
    return float(np.mean(tau_hat))


ESTIMATORS = {
    "Naive": naive_difference_in_means,
    "OLS": ols_with_controls,
    "IPW": ipw_estimator,
    "DML": dml_plr_estimator,
    "Causal Forest": causal_forest_t_learner,
}
