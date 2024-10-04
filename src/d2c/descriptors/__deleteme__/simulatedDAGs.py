import numpy as np
import networkx as nx
from networkx.algorithms.dag import is_directed_acyclic_graph
from multiprocessing import Pool
import pandas as pd
import random


from scipy.special import expit


from typing import List


from simulated import Simulated

class SimulatedDAGs(Simulated):

    def __init__(self, n_dags: int, n_observations: int, n_variables: int, not_acyclic: bool = False, n_jobs: int = 1, random_state: int = 42, function_types: list = ["linear"], sdn: int = 0.001):
        super().__init__(n_dags, n_observations, n_variables, not_acyclic = not_acyclic, n_jobs=n_jobs, random_state=random_state, function_types=function_types, sdn=sdn)

    def generate(self) -> None:
        """Generates ndag number of DAGs."""
        if self.n_jobs == 1:
            self.list_initial_dags = [self._generate_single_dag(i) for i in range(self.n_dags)]
            self.list_observations = [self._simulate_single_dag_observations(dag) for dag in self.list_initial_dags]
        else:
            with Pool(processes=self.n_jobs) as pool:
                self.list_initial_dags = pool.map(self._generate_single_dag, range(self.n_dags))
                self.list_observations = pool.map(self._simulate_single_dag_observations, self.list_initial_dags)



    def _simulate_single_dag_observations(self, dag: nx.DiGraph) -> pd.DataFrame:
        """
        Simulates observations for a single DAG.
        
        Args:
            dag (nx.DiGraph): The DAG for which to simulate observations.

        Returns:
            pd.DataFrame: A DataFrame containing the simulated observations.
        """
        data = pd.DataFrame(index=range(self.n_observations), columns=dag.nodes)

        for node in nx.topological_sort(dag):
            parents = list(dag.predecessors(node))
            node_data = dag.nodes[node]
            bias = node_data['bias']
            sigma = node_data['sigma']

            if not parents:
                data[node] = np.random.normal(loc=bias, scale=sigma, size=self.n_observations)
            else:
                data[node] = bias
                # if self.additive: #TODO: implement nonadditive cases
                for parent in parents:
                    edge_data = dag.edges[parent, node]
                    # weight = edge_data['weight'] #TODO: check weight implementation
                    H = edge_data['H']
                    if H == "linear":
                        a = np.random.uniform(-1, 1, 2).reshape(2, 1)
                        X = np.array([data[parent] ** i for i in range(2)])    # data[node] += a0 * 1 + a1 * data[parent] 
                        data[node] += np.sum(X * a, axis=0)
                    elif H == "quadratic":
                        a = np.random.uniform(-1, 1, 3).reshape(3, 1)
                        X = np.array([data[parent] ** i for i in range(3)])
                        data[node] += np.sum(X * a, axis=0)
                    elif H == "exponential":
                        a = np.random.uniform(-1, 1)
                        b = np.random.uniform(0, 1)
                        data[node] += a * np.exp(b * data[parent])
                    elif H == "logarithmic": #could capture a slowing or saturating effect.
                        a = np.random.uniform(-1, 1)
                        b = np.random.uniform(1, 2)
                        data[node] += a * np.log(b + data[parent])
                    elif H == "sigmoid": #could model a system that has a thresholding or saturating effect.
                        a = np.random.uniform(-5, 5)
                        b = np.random.uniform(0, 1)
                        data[node] += 1 / (1 + np.exp(-a * (data[parent] - b)))
                    elif H == "sinusoidal": #can model periodic effects
                        a = np.random.uniform(-1, 1)
                        b = np.random.uniform(0, 2 * np.pi)
                        data[node] += a * np.sin(b + data[parent])
                data[node] += np.random.normal(scale=sigma, size=self.n_observations)

        return data
    
    def get_observations(self) -> List[pd.DataFrame]:
        """Returns the simulated observations."""
        return self.list_observations

    def get_dags(self) -> List[nx.DiGraph]:
        """Returns the generated DAGs."""
        return self.list_initial_dags
    


import time

def main():
    n_dags = 300
    # Other parameters that might be needed for your class, adjust as needed
    n_observations = 10
    n_variables = 5
    
    # Testing with a single process
    start_time = time.time()
    generator = SimulatedDAGs(n_dags, n_observations, n_variables, n_jobs=1)
    generator.generate()
    end_time = time.time()
    print(f"Time taken with a single process: {end_time - start_time:.2f} seconds")

    # Testing with multiprocessing, up to 6 cores
    for cores in range(2, 7):
        start_time = time.time()
        generator = SimulatedDAGs(n_dags, n_observations, n_variables, n_jobs=cores)
        generator.generate()
        end_time = time.time()
        print(f"Time taken with {cores} cores: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
