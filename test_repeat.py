from src.evaluation import repeated_evaluation
if __name__ == '__main__':
    r=repeated_evaluation('simulation',n_runs=8,n=3000,seed=100)
    print('done', r.shape)
