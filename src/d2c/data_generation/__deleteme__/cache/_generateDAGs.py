import argparse
import sys
import numpy as np

import pickle
import multiprocessing as mp

# Add the sys.path appends
sys.path.append("..")
sys.path.append("../d2c/")

from descriptors.simulated import Simulated
from descriptors.simulatedDAGs import SimulatedDAGs

def generate_dags(n_dags, n_observations, n_variables, n_jobs=10, name='dag', random_state=None):
    generator = SimulatedDAGs(n_dags, n_observations, n_variables, n_jobs=n_jobs, random_state=random_state)
    generator.generate()

    observations = generator.get_observations()
    dags = generator.get_dags()

    #pickle everything
    with open(f'../data/{name}.pkl', 'wb') as f:
        pickle.dump((observations, dags), f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Simulated DAGs')
    parser.add_argument('--n_dags', type=int, default=20, help='Number of DAGs')
    parser.add_argument('--n_observations', type=int, default=150, help='Number of observations per DAG')
    parser.add_argument('--n_variables', type=int, default=6, help='Number of variables per observation')
    parser.add_argument('--n_jobs', type=int, default=10, help='Number of jobs for parallel processing')
    parser.add_argument('--name', type=str, default='dag', help='Name for the output file')
    parser.add_argument('--random_state', type=int, default=None, help='Random state for reproducibility')

    args = parser.parse_args()

    generate_dags(args.n_dags, args.n_observations, args.n_variables, args.n_jobs, args.name, args.random_state)
