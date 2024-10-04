import random
import math
import networkx as nx
import pickle
import pandas as pd
import numpy as np

#if from main
from src.data_generation.gen_utils import get_causal_dfs, custom_layout, show_DAG
from src.data_generation.knownts import *

# from gen_utils import get_causal_dfs, custom_layout, show_DAG
# from knownts import *

class TSBuilder():
    """

    """

    def __init__(self, timesteps=3, maxlags=3, n_variables=20, n_iterations=100, processes_to_use = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], noise_std=0.1, seed=42, verbose = True):
        self.timesteps = timesteps
        self.maxlags = maxlags
        self.processes_to_use = processes_to_use
        self.n_variables = n_variables
        self.n_iterations = n_iterations
        self.utils = get_utils(timesteps)
        self.verbose = verbose

        self.observations_list_dict = {}
        self.dags_dict = {}
        self.causal_dfs_dict = {}
        self.neighboors_dict = {}
        self.noise_std = noise_std
        self.seed = seed

        random.seed(seed)

    def build(self):

        for i in self.processes_to_use:
            if self.verbose: print('Generating data for process', i)

            dag_function, ts_function, range_t = self.utils[i]

            rename_dict = {f"Y[{t}][{j}]": f"{j}_t-{self.maxlags-t}" for t in range(self.maxlags+1) for j in range(self.n_variables)}

            observations_list, dags, causal_df, neighboors = [], [], [], []

            W = [[random.gauss(0, self.noise_std) for _ in range(self.n_variables)] for _ in range(self.timesteps+1)]

            causal_dfs = []
            for _ in range(self.n_iterations):
                valid_observations = False
                while not valid_observations:
                    size_N_j = [random.randint(1, 3) for j in range(self.n_variables)] #TODO: size of neighborhood is hardcoded

                    N_j = [random.sample(range(self.n_variables), size_N_j[j]) for j in range(self.n_variables)] #neighborhood of j, changes at each iteration
                    
                    Y_n = [[random.uniform(-1, 1) for _ in range(self.n_variables)] for _ in range(self.timesteps+1)]  #initialization of Y_n
                    if i == 16: #one of the processes has an exogenous variable
                        x = [1] + [0 for _ in range(self.timesteps)]

                    dag = dag_function(T=self.maxlags,N_j=N_j, N=self.n_variables) #get the dags from the corresponding functions
                    dag = nx.relabel_nodes(dag, rename_dict) #rename the nodes of the dag to use the same convention as the descriptors
                    causal_df = get_causal_dfs(dag, self.n_variables, self.maxlags) # get the causal_df from the dag

                    for t in range_t:
                        if i == 16: #one of the processes has an exogenous variable
                            for j in range(self.n_variables):
                                Y_n[t+1][j] = ts_function(Y_n, t, j, N_j[j], W, x[t]) #generate the observations according to the corresponding function
                            x[t+1] = update_x(x[t])

                        else:
                            for j in range(self.n_variables):
                                Y_n[t+1][j] = ts_function(Y_n, t, j, N_j[j], W) #generate the observations according to the corresponding function
                                
                    observations = pd.DataFrame(Y_n).replace([np.inf, -np.inf], np.nan)
                    
                    if observations.isnull().sum().sum() == 0:
                        valid_observations = True
                    else:
                        print(f"Found 'inf' or NAn in observations, reinitializing...")

                observations_list.append(observations)
                dags.append(dag)
                causal_dfs.append(causal_df)
                neighboors.append(N_j)

            self.observations_list_dict[i] = observations_list
            self.dags_dict[i] = dags
            self.causal_dfs_dict[i] = causal_dfs
            self.neighboors_dict[i] = neighboors

    def save(self, output_folder = 'data/synthetic', single_file = False):
        # if single_file:
        #     with open(output_folder + f'data.pkl', 'wb') as handle:
        #         pickle.dump((self.observations_list_dict, self.dags_dict, self.causal_dfs_dict, self.neighboors_dict), handle)
        for i in self.processes_to_use:
            with open(output_folder + f'data_{i}.pkl', 'wb') as handle:
                pickle.dump((self.observations_list_dict[i], self.dags_dict[i], self.causal_dfs_dict[i], self.neighboors_dict[i]), handle)

    def get_observations(self):
        return self.observations_list_dict
    
    def get_dags(self):
        return self.dags_dict
    
    def get_causal_dfs(self):
        return self.causal_dfs_dict

    def get_n_variables(self):
        return self.n_variables
    
    def get_maxlags(self):
        return self.maxlags

if __name__ == "__main__":
    
    
    T = 200  # Time steps
    maxlags = 3
    n_variables = 5   # number of j indices

    ts_builder = TSBuilder(timesteps=T, maxlags = maxlags, n_variables=n_variables, n_iterations=10)
    ts_builder.build()
