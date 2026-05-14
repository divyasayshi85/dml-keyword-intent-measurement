"""Run one estimator batch for subprocess-safe paper-mode execution."""
from __future__ import annotations
import argparse
from .data_generation import generate_simulated_data, generate_semi_synthetic_data
from .evaluation import estimate_all

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['simulation','semi_synthetic'], required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--confounding-strength', type=float, default=1.0)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.dataset == 'simulation':
        b = generate_simulated_data(n=args.n, confounding_strength=args.confounding_strength, seed=args.seed)
    else:
        b = generate_semi_synthetic_data(n=args.n, confounding_strength=args.confounding_strength, seed=args.seed)
    r = estimate_all(b, seed=args.seed)
    r['run'] = args.run
    r['confounding_strength'] = args.confounding_strength
    r.to_csv(args.output, index=False)

if __name__ == '__main__':
    main()
