import argparse
import sys
sys.path.append("..")
sys.path.append("../d2c/")
import numpy as np
from descriptors.simulated import Simulated
from descriptors.simulatedTimeSeries import SimulatedTimeSeries
import pickle

DIVERGENCE_THRESHOLD = 1e3

def check_divergence(multivariate_series):
    for i in range(multivariate_series.shape[1]):
        series = multivariate_series.iloc[:, i]
        if np.any(np.isnan(series)) or np.any(np.isinf(series)):
            return True  # Found NaN or Inf, indicating divergence
        if np.var(series) > DIVERGENCE_THRESHOLD:
            return True  # Variance is too high, indicating potential divergence
        if np.mean(np.abs(series)) > DIVERGENCE_THRESHOLD:
            return True  # Mean absolute value is too high, which might indicate divergence

def check_zero(multivariate_series):
    for i in range(multivariate_series.shape[1]):
        series = multivariate_series.iloc[:, i]
        if series.values[-1] == 0:
            return True   # Found Zero, indicating convergence to zero

def generate_time_series(n_series, n_observations, n_variables, maxlags, not_acyclic, n_jobs, noise, name, random_state, function_types):
    print(function_types)
    generator = SimulatedTimeSeries(n_series, n_observations, n_variables, not_acyclic=not_acyclic, maxlags=maxlags, n_jobs=n_jobs, sdn=noise, random_state=random_state, function_types=function_types)
    generator.generate()
    observations = generator.get_observations()
    dags = generator.get_dags()
    updated_dags = generator.get_updated_dags()
    causal_dfs = generator.get_causal_dfs()
    #pickle everything
    
    divergent_series_indices = []
    for obs_idx, obs in enumerate(observations):
        if check_divergence(obs):
            print(f'WARNING: Series {obs_idx} is divergent')
            divergent_series_indices.append(obs_idx)
            
        if check_zero(obs):
            print(f'WARNING: Series {obs_idx} is zero')
            divergent_series_indices.append(obs_idx)

    # Reverse sort indices and remove them from the end to avoid index shifting
    for idx in sorted(divergent_series_indices, reverse=True):
        observations.pop(idx)
        dags.pop(idx)
        updated_dags.pop(idx)
        causal_dfs.pop(idx)

    for obs_idx, obs in enumerate(observations): #TODO: check why this is necessary  despite previous check
        # drop nas
        observations[obs_idx] = obs.dropna()

    #print how many left
    print(f'Generated {len(observations)} time series')
    with open(f'../data/{name}.pkl', 'wb') as f:
        pickle.dump((observations, dags, updated_dags, causal_dfs), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Simulated Time Series')
    parser.add_argument('--n_series', type=int, default=100, help='Number of series')
    parser.add_argument('--n_observations', type=int, default=150, help='Number of observations per series')
    parser.add_argument('--n_variables', type=int, default=3, help='Number of variables per observation')
    parser.add_argument('--maxlags', type=int, default=3, help='Maximum lags for the time series')
    parser.add_argument('--not_acyclic', type=bool, default=True, help='Whether the DAGs should allow cyclic cause-effect pairs when looking at the past')
    parser.add_argument('--n_jobs', type=int, default=10, help='Number of jobs for parallel processing')
    parser.add_argument('--noise', type=float, default=0.1, help='Noise level for the time series')
    parser.add_argument('--name', type=str, default='ts3', help='Name of the file to save the data')
    parser.add_argument('--random_state', type=int, default=0, help='Random state for reproducibility')
    parser.add_argument('--function_types', nargs='+', default=['sigmoid','linear','quadratic','exponential','tanh','polynomial'], help='Types of functions to use in the time series')
    args = parser.parse_args()

    generate_time_series(args.n_series, args.n_observations, args.n_variables, args.maxlags, args.not_acyclic, args.n_jobs, args.noise, args.name, args.random_state, args.function_types)
