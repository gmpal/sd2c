"""
This module is responsible for generating data based on the specified parameters.
"""

from typing import List, Dict
from CausalPlayground import SCMGenerator
import random
from d2c.data_generation.functions import (
    f_linear,
    polynomial_factory,
    f_sigmoid,
    nonlinear_factory,
    f_interaction,
)
import random
import pandas as pd
import networkx as nx
import pickle

from typing import List, Dict


class Builder:
    """
    Builder class for generating synthetic datasets and corresponding directed acyclic graphs (DAGs) using specified functions.
    Attributes:
        functions_to_use (List[str]): List of functions to use for generating data.
        noise_std (float): Standard deviation of the noise (currently not used).
        generated_observations (Dict[str, List[pd.DataFrame]]): Dictionary where the key is the function name and the value is a list of dataframes containing generated observations.
        generated_dags (Dict[str, List[nx.DiGraph]]): Dictionary where the key is the function name and the value is a list of generated DAGs.
    Methods:
        __init__(self, observations: int = 200, n_variables_exo: int = 5, n_variables_endo: int = 5, functions_to_use: List[str] = ["linear", "polynomial", "sigmoid", "nonlinear", "interaction"], datasets_per_function: int = 10, noise_std: float = 0.1, seed: int = 42):
            Initializes the Builder class with the specified parameters.
        build(self):
            Generates synthetic datasets and corresponding DAGs using the specified functions and parameters.
        to_pickle(self, path: str):
            Saves the generated data (observations and DAGs) to a pickle file at the specified path.
    """

    def __init__(
        self,
        observations: int = 200,
        n_variables_exo: int = 5,
        n_variables_endo: int = 5,
        functions_to_use: List[str] = [
            "linear",
            "polynomial",
            "sigmoid",
            "nonlinear",
            "interaction",
        ],
        functions_kwargs: Dict[str, List] = {}, #eg {'polynomial': [[1, 2], [2, 3]], 'nonlinear': [math.sin, math.cos]}
        datasets_per_function: int = 10,
        noise_std: float = 0.1,  # currently not used #TODO: include
        seed: int = 42,
    ) -> None:
        """
        Initializes the Builder class.

        Args:
            observations (int): Number of observations per time series.
            n_variables_exo (int): Number of exogenous variables.
            n_variables_endo (int): Number of endogenous variables.
            functions_to_use (List[str]): List of functions to use.
            datasets_per_function (int): Number of datasets to generate per function.
            noise_std (float): Standard deviation of the noise.
            seed (int): Seed for reproducibility.
        """
        self.observations = observations
        self.n_variables_exo = n_variables_exo
        self.n_variables_endo = n_variables_endo
        self.datasets_per_function = datasets_per_function
        self.functions_to_use = functions_to_use
        self.functions_kwargs = functions_kwargs
        self.noise_std = noise_std  # currently not used #TODO: include
        self.seed = seed

        self.generated_observations: Dict[str, List[pd.DataFrame]] = (
            {}
        )  # dictionary where key is function and value is List[dataframes]
        self.generated_dags: Dict[str, List[nx.DiGraph]] = {}


        # Create a dictionary to hold your polynomial functions
        self.polynomial_functions = {}
        # Generate the polynomial functions and store them in the dictionary
        for degrees in functions_kwargs['polynomial']:
            func_name = f"poly_deg_{'_'.join(map(str, degrees))}"
            self.polynomial_functions[func_name] = polynomial_factory(degrees)

        # Create a dictionary to hold your nonlinear functions
        self.nonlinear_functions = {}
        # Generate the nonlinear functions and store them in the dictionary
        for nonlinearity in functions_kwargs['nonlinear']:
            func_name = f"poly_deg_{'_'.join(map(str, nonlinearity))}"
            self.nonlinear_functions[func_name] = nonlinear_factory(nonlinearity)




    def build(self) -> None:
        """
        Generates synthetic datasets and corresponding directed acyclic graphs (DAGs) using specified functions.
        This method iterates over the functions specified in `self.functions_to_use` and generates a specified number
        of datasets and DAGs for each function. The synthetic data is generated using the `SCMGenerator` class, which
        creates structural causal models (SCMs) with random parameters.
        The generated datasets and DAGs are stored in `self.generated_observations` and `self.generated_dags`
        dictionaries, respectively, with the function names as keys.
        Parameters:
        None
        Returns:
        None
        """

        gen = SCMGenerator(
            all_functions={
                "linear": f_linear,
                "polynomial": f_polynomial,
                "sigmoid": f_sigmoid,
                "nonlinear": f_nonlinear,
                "interaction": f_interaction,
            },
            seed=0,
        )

        for function_to_use in self.functions_to_use:

            list_of_observations_df = []
            list_of_dags = []

            for _ in range(self.datasets_per_function):

                scm = gen.create_random(
                    possible_functions=[function_to_use],
                    n_endo=self.n_variables_endo,
                    n_exo=self.n_variables_exo,
                    exo_distribution=random.random,  # TODO: allow user to change
                    exo_distribution_kwargs={},  # TODO: allow user to change
                )[0]

                samples_list = []
                for _ in range(self.observations):
                    endogenous_vars_dict, exogenous_vars_dict = scm.get_next_sample()
                    # make a single dictionary
                    full_vars_dict = {**endogenous_vars_dict, **exogenous_vars_dict}
                    # observations_list.append(full_vars_dict)
                    samples_list.append(full_vars_dict)

                observations_df = pd.DataFrame(samples_list)
                dag = scm.create_graph()

                list_of_observations_df.append(observations_df)
                list_of_dags.append(dag)

            self.generated_observations[function_to_use] = list_of_observations_df
            self.generated_dags[function_to_use] = list_of_dags

    def get_generated_observations(self) -> Dict[str, List[pd.DataFrame]]:
        return self.generated_observations

    def get_generated_dags(self) -> Dict[str, List[nx.DiGraph]]:
        return self.generated_dags

    def to_pickle(self, path) -> None:
        """
        Saves the generated data to a pickle file.

        Args:
            path (str): Path to save the data.
        """
        with open(path, "wb") as f:
            pickle.dump(
                (
                    self.generated_observations,
                    self.generated_dags,
                ),
                f,
            )
