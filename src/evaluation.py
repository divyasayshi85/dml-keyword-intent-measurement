"""Evaluation metrics and robustness checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_generation import DatasetBundle, generate_semi_synthetic_data, generate_simulated_data
from .estimators import ESTIMATORS, propensity_scores


def estimate_all(bundle: DatasetBundle, seed: int = 42) -> pd.DataFrame:
    """Estimate ATE using all estimators."""
    rows = []
    for method_name, estimator in ESTIMATORS.items():
        estimated_ate = estimator(bundle.X, bundle.treatment, bundle.outcome, seed=seed)
        rows.append({
            "dataset": bundle.name,
            "method": method_name,
            "estimated_ate": estimated_ate,
            "true_ate": bundle.true_ate,
            "bias": estimated_ate - bundle.true_ate,
            "absolute_bias": abs(estimated_ate - bundle.true_ate),
            "relative_error_pct": 100.0 * (estimated_ate - bundle.true_ate) / bundle.true_ate,
        })
    return pd.DataFrame(rows)




def repeated_evaluation(
    dataset_type: str,
    n_runs: int = 8,
    n: int = 3000,
    confounding_strength: float = 1.0,
    seed: int = 1000,
) -> pd.DataFrame:
    """Repeat estimation across seeds in-process.

    The main run_all.py script uses subprocess-safe execution for paper mode;
    this function remains available for lightweight interactive use.
    """
    frames = []
    for run in range(n_runs):
        run_seed = seed + run
        if dataset_type == "simulation":
            bundle = generate_simulated_data(n=n, confounding_strength=confounding_strength, seed=run_seed)
        elif dataset_type == "semi_synthetic":
            bundle = generate_semi_synthetic_data(n=n, confounding_strength=confounding_strength, seed=run_seed)
        else:
            raise ValueError("dataset_type must be 'simulation' or 'semi_synthetic'.")
        result = estimate_all(bundle, seed=run_seed)
        result["run"] = run
        frames.append(result)
    return pd.concat(frames, ignore_index=True)

def summarize_performance(results: pd.DataFrame) -> pd.DataFrame:
    """Compute summary metrics by dataset and method."""
    summary = results.groupby(["dataset", "method"], as_index=False).agg(
        mean_estimated_ate=("estimated_ate", "mean"),
        true_ate=("true_ate", "mean"),
        bias=("bias", "mean"),
        absolute_bias=("absolute_bias", "mean"),
        variance=("estimated_ate", "var"),
        rmse=("bias", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        mean_relative_error_pct=("relative_error_pct", "mean"),
    )
    return summary.sort_values(["dataset", "rmse"]).reset_index(drop=True)


def placebo_test(bundle: DatasetBundle, seed: int = 42) -> pd.DataFrame:
    """Permute treatment to create a zero-effect placebo check."""
    rng = np.random.default_rng(seed)
    placebo_bundle = DatasetBundle(
        X=bundle.X,
        treatment=rng.permutation(bundle.treatment),
        outcome=bundle.outcome,
        true_tau=np.zeros_like(bundle.true_tau),
        latent_u=bundle.latent_u,
        name=f"{bundle.name}_placebo",
    )

    rows = []
    for method_name, estimator in ESTIMATORS.items():
        estimated = estimator(placebo_bundle.X, placebo_bundle.treatment, placebo_bundle.outcome, seed=seed)
        rows.append({
            "dataset": placebo_bundle.name,
            "method": method_name,
            "placebo_estimated_ate": estimated,
            "target_effect": 0.0,
            "absolute_placebo_effect": abs(estimated),
        })
    return pd.DataFrame(rows)


def confounding_sensitivity(
    dataset_type: str = "simulation",
    strengths: np.ndarray | None = None,
    n: int = 3000,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate absolute bias as latent confounding increases."""
    if strengths is None:
        strengths = np.linspace(0, 1.5, 5)

    frames = []
    for idx, strength in enumerate(strengths):
        if dataset_type == "simulation":
            bundle = generate_simulated_data(n=n, confounding_strength=float(strength), seed=seed + idx)
        elif dataset_type == "semi_synthetic":
            bundle = generate_semi_synthetic_data(n=n, confounding_strength=float(strength), seed=seed + idx)
        else:
            raise ValueError("dataset_type must be 'simulation' or 'semi_synthetic'.")

        result = estimate_all(bundle, seed=seed + idx)
        result["confounding_strength"] = float(strength)
        frames.append(result)

    return pd.concat(frames, ignore_index=True)


def sample_size_sensitivity(
    dataset_type: str = "simulation",
    sample_sizes: list[int] | None = None,
    n_runs: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate RMSE across sample sizes."""
    if sample_sizes is None:
        sample_sizes = [1000, 2000, 3000, 5000]

    frames = []
    for n in sample_sizes:
        repeated = repeated_evaluation(dataset_type, n_runs=n_runs, n=n, seed=seed + n)
        summary = summarize_performance(repeated)
        summary["sample_size"] = n
        frames.append(summary)

    return pd.concat(frames, ignore_index=True)


def overlap_diagnostics(bundle: DatasetBundle) -> pd.DataFrame:
    """Return estimated propensity scores by treatment group."""
    return pd.DataFrame({
        "dataset": bundle.name,
        "treatment": bundle.treatment,
        "propensity_score": propensity_scores(bundle.X, bundle.treatment),
    })
