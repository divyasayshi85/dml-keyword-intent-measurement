from __future__ import annotations
import os, time
import pandas as pd
from src.allocation import allocation_efficiency
from src.data_generation import generate_semi_synthetic_data, generate_simulated_data
from src.evaluation import confounding_sensitivity, overlap_diagnostics, placebo_test, repeated_evaluation, sample_size_sensitivity, summarize_performance
from src.figures import plot_allocation_efficiency, plot_bias_vs_confounding, plot_estimator_comparison, plot_heterogeneous_effects, plot_overlap, plot_sample_size_rmse, plot_selection_bias_mechanism, set_publication_style
OUTPUT_DIR='outputs'; FIGURE_DIR=os.path.join(OUTPUT_DIR,'figures'); TABLE_DIR=os.path.join(OUTPUT_DIR,'tables')
def mark(s,t0): print(f'{s}: {time.time()-t0:.1f}s', flush=True)
def main():
    t0=time.time(); os.makedirs(FIGURE_DIR,exist_ok=True); os.makedirs(TABLE_DIR,exist_ok=True); set_publication_style(); mark('init',t0)
    simulation=generate_simulated_data(n=3000,confounding_strength=1.0,seed=42); semi_synthetic=generate_semi_synthetic_data(n=3000,confounding_strength=1.0,seed=123); mark('datasets',t0)
    sim_results=repeated_evaluation('simulation',n_runs=8,n=3000,confounding_strength=1.0,seed=100); mark('sim repeated',t0)
    semi_results=repeated_evaluation('semi_synthetic',n_runs=8,n=3000,confounding_strength=1.0,seed=300); mark('semi repeated',t0)
    all_results=pd.concat([sim_results,semi_results],ignore_index=True); all_results.to_csv(os.path.join(TABLE_DIR,'estimator_results_by_run.csv'),index=False); summarize_performance(all_results).to_csv(os.path.join(TABLE_DIR,'table1_estimator_performance.csv'),index=False); mark('performance saved',t0)
    placebo=pd.concat([placebo_test(simulation,seed=500),placebo_test(semi_synthetic,seed=600)],ignore_index=True); placebo.to_csv(os.path.join(TABLE_DIR,'placebo_test_results.csv'),index=False); mark('placebo',t0)
    confounding=pd.concat([confounding_sensitivity('simulation',n=3000,seed=700),confounding_sensitivity('semi_synthetic',n=3000,seed=900)],ignore_index=True); confounding.to_csv(os.path.join(TABLE_DIR,'confounding_sensitivity.csv'),index=False); mark('confounding',t0)
    sample_sizes=sample_size_sensitivity('simulation',n_runs=4,sample_sizes=[1000,2000,3000,5000],seed=1100); sample_sizes.to_csv(os.path.join(TABLE_DIR,'sample_size_sensitivity.csv'),index=False); mark('sample sizes',t0)
    overlap=pd.concat([overlap_diagnostics(simulation),overlap_diagnostics(semi_synthetic)],ignore_index=True); overlap.to_csv(os.path.join(TABLE_DIR,'overlap_diagnostics.csv'),index=False); mark('overlap',t0)
    allocation=allocation_efficiency(simulation,top_fraction=0.25,seed=1200); allocation.to_csv(os.path.join(TABLE_DIR,'allocation_efficiency.csv'),index=False); mark('allocation',t0)
    plot_selection_bias_mechanism(os.path.join(FIGURE_DIR,'figure1_selection_bias_mechanism.png')); plot_estimator_comparison(sim_results,os.path.join(FIGURE_DIR,'figure2_estimator_comparison.png')); plot_bias_vs_confounding(confounding[confounding['dataset']=='simulation'],os.path.join(FIGURE_DIR,'figure3_bias_vs_confounding.png')); plot_sample_size_rmse(sample_sizes,os.path.join(FIGURE_DIR,'figure4_sample_size_rmse.png')); plot_heterogeneous_effects(simulation.true_tau,os.path.join(FIGURE_DIR,'figure5_heterogeneous_effects.png')); plot_allocation_efficiency(allocation,os.path.join(FIGURE_DIR,'figure6_allocation_efficiency.png')); plot_overlap(overlap[overlap['dataset']=='simulation'],os.path.join(FIGURE_DIR,'appendix_overlap_diagnostics.png')); mark('figures',t0)
    print('Analysis complete.')
if __name__=='__main__': main()
