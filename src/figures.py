"""Publication-ready figure generation."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_publication_style() -> None:
    """Apply consistent publication style."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def savefig(path: str) -> None:
    """Save and close current figure."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_selection_bias_mechanism(path: str) -> None:
    """Figure 1: conceptual selection bias mechanism."""
    x = np.linspace(0, 1, 250)
    sigmoid = lambda z: 1 / (1 + np.exp(-z))

    plt.figure(figsize=(7, 4.8))
    plt.plot(x, sigmoid(6 * (x - 0.42)), linewidth=2.2, label="P(Treatment | U)")
    plt.plot(x, sigmoid(6 * (x - 0.30)), linewidth=2.2, label="P(Outcome | U)")
    plt.xlabel("Latent Propensity (U)")
    plt.ylabel("Probability")
    plt.title("Selection Bias Mechanism")
    plt.legend(frameon=False)
    savefig(path)


def plot_estimator_comparison(results: pd.DataFrame, path: str) -> None:
    """Figure 2: estimator comparison with 95% interval."""
    order = ["Naive", "OLS", "IPW", "DML", "Causal Forest"]
    summary = results.groupby("method", as_index=False).agg(
        estimated_ate=("estimated_ate", "mean"),
        se=("estimated_ate", "std"),
        true_ate=("true_ate", "mean"),
    ).fillna(0)
    summary = summary.set_index("method").loc[order].reset_index()

    x = np.arange(len(order))
    plt.figure(figsize=(8, 4.8))
    plt.errorbar(
        x,
        summary["estimated_ate"],
        yerr=1.96 * summary["se"],
        fmt="o",
        capsize=5,
        label="Estimated ATE",
    )
    plt.axhline(float(summary["true_ate"].iloc[0]), linestyle="--", label="True Effect")
    plt.xticks(x, order)
    plt.ylabel("Estimated Treatment Effect")
    plt.title("Estimator Comparison vs True Effect")
    plt.legend(frameon=False)
    savefig(path)


def plot_bias_vs_confounding(df: pd.DataFrame, path: str) -> None:
    """Figure 3: absolute bias by confounding strength."""
    plt.figure(figsize=(8, 4.8))
    for method, method_df in df.groupby("method"):
        method_df = method_df.sort_values("confounding_strength")
        plt.plot(
            method_df["confounding_strength"],
            method_df["absolute_bias"],
            marker="o",
            linewidth=2,
            label=method,
        )
    plt.xlabel("Confounding Strength")
    plt.ylabel("Absolute Bias")
    plt.title("Bias vs Confounding Strength")
    plt.legend(frameon=False)
    savefig(path)


def plot_sample_size_rmse(df: pd.DataFrame, path: str) -> None:
    """Figure 4: RMSE by sample size."""
    plt.figure(figsize=(8, 4.8))
    for method, method_df in df.groupby("method"):
        method_df = method_df.sort_values("sample_size")
        plt.plot(
            method_df["sample_size"],
            method_df["rmse"],
            marker="o",
            linewidth=2,
            label=method,
        )
    plt.xlabel("Sample Size")
    plt.ylabel("RMSE")
    plt.title("Estimation Error vs Sample Size")
    plt.legend(frameon=False)
    savefig(path)


def plot_heterogeneous_effects(true_tau: np.ndarray, path: str) -> None:
    """Figure 5: distribution of true heterogeneous effects."""
    plt.figure(figsize=(7, 4.8))
    plt.hist(true_tau, bins=30, edgecolor="black", alpha=0.85)
    plt.xlabel("Treatment Effect")
    plt.ylabel("Frequency")
    plt.title("Distribution of Heterogeneous Treatment Effects")
    savefig(path)


def plot_allocation_efficiency(df: pd.DataFrame, path: str) -> None:
    """Figure 6: allocation efficiency by method."""
    order = ["Naive", "OLS", "IPW", "Causal Forest", "DML", "Optimal"]
    plot_df = df.set_index("method").loc[order].reset_index()

    plt.figure(figsize=(8, 4.8))
    plt.bar(np.arange(len(order)), plot_df["relative_efficiency"])
    plt.xticks(np.arange(len(order)), order)
    plt.ylabel("Relative Outcome Efficiency")
    plt.ylim(0, 1.08)
    plt.title("Allocation Efficiency Across Methods")
    savefig(path)


def plot_overlap(df: pd.DataFrame, path: str) -> None:
    """Appendix figure: propensity-score overlap."""
    plt.figure(figsize=(7, 4.8))
    plt.hist(df.loc[df["treatment"] == 0, "propensity_score"], bins=25, alpha=0.65, label="Control")
    plt.hist(df.loc[df["treatment"] == 1, "propensity_score"], bins=25, alpha=0.65, label="Treated")
    plt.xlabel("Estimated Propensity Score")
    plt.ylabel("Frequency")
    plt.title("Overlap Diagnostics")
    plt.legend(frameon=False)
    savefig(path)
