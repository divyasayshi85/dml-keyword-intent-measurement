"""Run the complete reproducible analysis in paper mode."""
from __future__ import annotations
import os, sys, subprocess, tempfile, time, json
import pandas as pd
from src.allocation import allocation_efficiency
from src.data_generation import generate_semi_synthetic_data, generate_simulated_data
from src.evaluation import summarize_performance, placebo_test, overlap_diagnostics
from src.figures import set_publication_style, plot_selection_bias_mechanism, plot_estimator_comparison, plot_bias_vs_confounding, plot_sample_size_rmse, plot_heterogeneous_effects, plot_allocation_efficiency, plot_overlap

OUTPUT_DIR='outputs'; FIGURE_DIR=os.path.join(OUTPUT_DIR,'figures'); TABLE_DIR=os.path.join(OUTPUT_DIR,'tables')
PAPER_N=3000; MAIN_RUNS=8; TREES=100; DML_FOLDS=5

def run_single(dataset, n, seed, run, confounding_strength=1.0, tmpdir=None):
    if tmpdir is None: tmpdir=tempfile.mkdtemp()
    out=os.path.join(tmpdir, f'{dataset}_run{run}_seed{seed}_conf{confounding_strength:.3f}.csv')
    cmd=[sys.executable,'-m','src.run_one','--dataset',dataset,'--n',str(n),'--seed',str(seed),'--run',str(run),'--confounding-strength',str(confounding_strength),'--output',out]
    proc = subprocess.Popen(cmd, cwd=os.getcwd(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    # Emit a heartbeat to avoid silent long-running periods in constrained execution environments.
    while proc.poll() is None:
        time.sleep(5)
        print('.', end='', flush=True)
    stderr = proc.stderr.read() if proc.stderr else ''
    print('', flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f'run_one failed for {dataset}, seed={seed}: {stderr}')
    return pd.read_csv(out)

def repeated_subprocess(dataset, n_runs, n, seed, confounding_strength=1.0):
    frames=[]
    with tempfile.TemporaryDirectory() as tmpdir:
        for run in range(n_runs):
            frames.append(run_single(dataset,n,seed+run,run,confounding_strength,tmpdir))
            print(f'{dataset} run {run+1}/{n_runs} complete', flush=True)
    return pd.concat(frames, ignore_index=True)

def confounding_subprocess(dataset, n, seed, strengths):
    frames=[]
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, strength in enumerate(strengths):
            frames.append(run_single(dataset,n,seed+i,i,float(strength),tmpdir))
            print(f'{dataset} confounding {i+1}/{len(strengths)} complete', flush=True)
    return pd.concat(frames, ignore_index=True)

def sample_size_subprocess(dataset, sample_sizes, n_runs, seed):
    frames=[]
    for n in sample_sizes:
        r=repeated_subprocess(dataset,n_runs,n,seed+n,1.0)
        s=summarize_performance(r); s['sample_size']=n; frames.append(s)
        print(f'{dataset} sample size {n} complete', flush=True)
    return pd.concat(frames, ignore_index=True)

def main():
    start=time.time()
    os.makedirs(FIGURE_DIR, exist_ok=True); os.makedirs(TABLE_DIR, exist_ok=True); set_publication_style()
    simulation=generate_simulated_data(n=PAPER_N, confounding_strength=1.0, seed=42)
    semi_synthetic=generate_semi_synthetic_data(n=PAPER_N, confounding_strength=1.0, seed=123)
    sim_results=repeated_subprocess('simulation', MAIN_RUNS, PAPER_N, 100)
    semi_results=repeated_subprocess('semi_synthetic', MAIN_RUNS, PAPER_N, 300)
    all_results=pd.concat([sim_results, semi_results], ignore_index=True)
    all_results.to_csv(os.path.join(TABLE_DIR,'estimator_results_by_run.csv'), index=False)
    summarize_performance(all_results).to_csv(os.path.join(TABLE_DIR,'table1_estimator_performance.csv'), index=False)
    pd.concat([placebo_test(simulation,seed=500), placebo_test(semi_synthetic,seed=600)], ignore_index=True).to_csv(os.path.join(TABLE_DIR,'placebo_test_results.csv'), index=False)
    strengths=[0.0,0.375,0.75,1.125,1.5]
    confounding=pd.concat([confounding_subprocess('simulation',PAPER_N,700,strengths), confounding_subprocess('semi_synthetic',PAPER_N,900,strengths)], ignore_index=True)
    confounding.to_csv(os.path.join(TABLE_DIR,'confounding_sensitivity.csv'), index=False)
    sample_sizes=sample_size_subprocess('simulation',[1000,2000,3000,5000],4,1100)
    sample_sizes.to_csv(os.path.join(TABLE_DIR,'sample_size_sensitivity.csv'), index=False)
    overlap=pd.concat([overlap_diagnostics(simulation), overlap_diagnostics(semi_synthetic)], ignore_index=True)
    overlap.to_csv(os.path.join(TABLE_DIR,'overlap_diagnostics.csv'), index=False)
    allocation=allocation_efficiency(simulation, top_fraction=0.25, seed=1200)
    allocation.to_csv(os.path.join(TABLE_DIR,'allocation_efficiency.csv'), index=False)
    plot_selection_bias_mechanism(os.path.join(FIGURE_DIR,'figure1_selection_bias_mechanism.png'))
    plot_estimator_comparison(sim_results, os.path.join(FIGURE_DIR,'figure2_estimator_comparison.png'))
    plot_bias_vs_confounding(confounding[confounding['dataset']=='simulation'], os.path.join(FIGURE_DIR,'figure3_bias_vs_confounding.png'))
    plot_sample_size_rmse(sample_sizes, os.path.join(FIGURE_DIR,'figure4_sample_size_rmse.png'))
    plot_heterogeneous_effects(simulation.true_tau, os.path.join(FIGURE_DIR,'figure5_heterogeneous_effects.png'))
    plot_allocation_efficiency(allocation, os.path.join(FIGURE_DIR,'figure6_allocation_efficiency.png'))
    plot_overlap(overlap[overlap['dataset']=='simulation'], os.path.join(FIGURE_DIR,'appendix_overlap_diagnostics.png'))
    elapsed=time.time()-start
    report={'runtime_seconds':elapsed,'runtime_minutes':elapsed/60,'configuration':{'main_n':PAPER_N,'main_runs':MAIN_RUNS,'trees':TREES,'dml_folds':DML_FOLDS,'sample_sizes':[1000,2000,3000,5000]}}
    with open(os.path.join(OUTPUT_DIR,'runtime_report.json'),'w') as f: json.dump(report,f,indent=2)
    print('Analysis complete.')
    print(f'Runtime seconds: {elapsed:.2f}')
    print(f'Runtime minutes: {elapsed/60:.2f}')
    print(f'Tables saved to: {TABLE_DIR}')
    print(f'Figures saved to: {FIGURE_DIR}')
if __name__=='__main__': main()
