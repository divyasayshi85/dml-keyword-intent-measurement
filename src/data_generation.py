"""IP-safe data generation for simulated and semi-synthetic experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


@dataclass(frozen=True)
class DatasetBundle:
    """Container for generated observational data."""
    X: pd.DataFrame
    treatment: np.ndarray
    outcome: np.ndarray
    true_tau: np.ndarray
    latent_u: np.ndarray
    name: str

    @property
    def true_ate(self) -> float:
        """Ground-truth average treatment effect."""
        return float(np.mean(self.true_tau))


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def generate_simulated_data(
    n: int = 1200,
    n_features: int = 8,
    confounding_strength: float = 1.0,
    seed: int = 42,
) -> DatasetBundle:
    """Generate a fully simulated observational dataset.

    The latent variable U affects both treatment assignment and outcomes,
    creating selection bias by construction.
    """
    rng = np.random.default_rng(seed)
    latent_u = rng.normal(size=n)
    noise = rng.normal(size=(n, n_features))

    X_raw = np.column_stack([
        0.55 * latent_u
        + 0.25 * np.sin(latent_u * (j + 1) / 3)
        + 0.20 * noise[:, j]
        for j in range(n_features)
    ])
    X = pd.DataFrame(X_raw, columns=[f"x{j+1}" for j in range(n_features)])

    nonlinear_x = (
        0.45 * X["x1"]
        - 0.35 * X["x2"]
        + 0.25 * X["x3"] ** 2
        - 0.20 * X["x4"] * X["x5"]
    )

    treatment_prob = sigmoid(
        -0.25
        + confounding_strength * 0.95 * latent_u
        + 0.65 * nonlinear_x.to_numpy()
    )
    treatment = rng.binomial(1, treatment_prob)

    true_tau = (
        0.030
        + 0.010 * sigmoid(X["x1"].to_numpy())
        - 0.006 * sigmoid(-X["x2"].to_numpy())
        + 0.004 * (X["x3"].to_numpy() > np.median(X["x3"].to_numpy()))
    )
    true_tau = np.clip(true_tau, 0.005, 0.075)

    baseline = sigmoid(
        -1.25
        + confounding_strength * 0.90 * latent_u
        + 0.35 * X["x1"].to_numpy()
        - 0.25 * X["x2"].to_numpy()
        + 0.15 * X["x3"].to_numpy() ** 2
    )

    outcome = np.clip(baseline + true_tau * treatment + rng.normal(0, 0.08, n), 0, 1)

    return DatasetBundle(X, treatment, outcome, true_tau, latent_u, "simulation")


def generate_semi_synthetic_data(
    n: int = 1200,
    n_features: int = 10,
    confounding_strength: float = 1.0,
    seed: int = 123,
) -> DatasetBundle:
    """Generate a semi-synthetic dataset from public-style feature distributions.

    The feature matrix is generated using a standard public sklearn generator.
    Treatment and outcomes are generated from a known structural model.
    """
    rng = np.random.default_rng(seed)

    X_base, y_latent = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=2,
        class_sep=1.0,
        random_state=seed,
    )
    X_base = (X_base - X_base.mean(axis=0)) / (X_base.std(axis=0) + 1e-8)

    latent_u = 0.7 * (y_latent - y_latent.mean()) + rng.normal(0, 0.7, n)
    X = pd.DataFrame(X_base, columns=[f"x{j+1}" for j in range(n_features)])

    nonlinear_x = (
        0.40 * X["x1"]
        + 0.30 * np.tanh(X["x2"])
        - 0.25 * X["x3"] * X["x4"]
        + 0.20 * X["x5"] ** 2
    )

    treatment_prob = sigmoid(
        -0.15
        + confounding_strength * 0.85 * latent_u
        + 0.55 * nonlinear_x.to_numpy()
    )
    treatment = rng.binomial(1, treatment_prob)

    true_tau = (
        0.028
        + 0.012 * sigmoid(X["x1"].to_numpy() + X["x2"].to_numpy())
        + 0.005 * (X["x6"].to_numpy() > 0)
    )
    true_tau = np.clip(true_tau, 0.005, 0.080)

    baseline = sigmoid(
        -1.10
        + confounding_strength * 0.80 * latent_u
        + 0.30 * X["x1"].to_numpy()
        + 0.20 * X["x2"].to_numpy()
        - 0.20 * X["x3"].to_numpy()
    )

    outcome = np.clip(baseline + true_tau * treatment + rng.normal(0, 0.08, n), 0, 1)

    return DatasetBundle(X, treatment, outcome, true_tau, latent_u, "semi_synthetic")
