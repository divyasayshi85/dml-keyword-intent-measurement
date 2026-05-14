# ARF DML Selection Bias Reproducibility Repository

Reproducible, IP-safe code for the paper:

**Correcting Selection Bias in Behavioral Intent Measurement Using Double Machine Learning**

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── run_all.py
├── src/
│   ├── __init__.py
│   ├── allocation.py
│   ├── data_generation.py
│   ├── estimators.py
│   ├── evaluation.py
│   └── figures.py
└── outputs/
    ├── figures/
    └── tables/
```

## Included

- Fully simulated data generation
- Semi-synthetic data generation using public-style feature distributions
- Five estimators:
  1. Naive difference-in-means
  2. OLS with controls
  3. Inverse Probability Weighting (IPW)
  4. Double Machine Learning (DML)
  5. Causal Forest-style T-learner benchmark
- Evaluation metrics:
  - bias
  - absolute bias
  - variance
  - RMSE
  - relative error
- Robustness checks:
  - placebo test
  - sensitivity to unobserved confounding
  - overlap diagnostics
- Allocation efficiency simulation
- Publication-ready figures
- CSV output tables

## Quick start

```bash
pip install -r requirements.txt
python run_all.py
```

Outputs are written to:

```text
outputs/figures/
outputs/tables/
```

## Notes on the causal forest benchmark

The causal forest component is implemented as a public, reproducible T-learner benchmark using random forests. It is included as a heterogeneous treatment effect benchmark and is not intended to represent any proprietary implementation.

## IP-safety statement

This repository uses only simulated and semi-synthetic data. It does not use proprietary datasets, internal systems, confidential algorithms, employer-specific logic, or platform-specific implementation details.


## Paper-mode configuration used in this package

This checked version uses a stronger publication-oriented configuration:

- main dataset size: `n = 3000`
- repeated runs: `n_runs = 8` for the main simulation and semi-synthetic comparisons
- DML nuisance models: `120` trees
- causal forest-style benchmark: `120` trees
- DML folds: `5`
- sample-size sensitivity: `1000`, `2000`, `3000`, `5000`

This is slower than the fast test configuration, but produces more stable paper-ready figures and tables.
