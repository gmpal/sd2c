"""
This module is responsible for generating data based on the specified parameters.
"""

from typing import List, Dict
from CausalPlayground import SCMGenerator
import random
from .functions import (
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
        functions_kwargs: Dict[
            str, List
        ] = {},  # eg {'polynomial': [[1, 2], [2, 3]], 'nonlinear': [math.sin, math.cos]}
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
        if "polynomial" in functions_kwargs:
            for degrees in functions_kwargs["polynomial"]:
                func_name = f"poly_deg_{'_'.join(map(str, degrees))}"
                self.polynomial_functions[func_name] = polynomial_factory(degrees)

        # Create a dictionary to hold your nonlinear functions
        self.nonlinear_functions = {}
        # Generate the nonlinear functions and store them in the dictionary
        if "nonlinear" in functions_kwargs:
            for nonlinearity in functions_kwargs["nonlinear"]:
                func_name = f"nonlinear_{nonlinearity.__name__}"
                self.nonlinear_functions[func_name] = nonlinear_factory(nonlinearity)

    def _get_all_possible_functions(self) -> Dict[str, callable]:
        """
        Retrieve all possible functions available for generation.
        This method aggregates various types of functions including linear,
        sigmoid, interaction, polynomial, and nonlinear functions into a single
        dictionary. The keys of the dictionary are the names of the functions,
        and the values are the corresponding callable function objects.
        Returns:
            Dict[str, callable]: A dictionary containing all possible functions
            with their names as keys and callable objects as values.
        """

        all_functions = {
            "linear": f_linear,
            "sigmoid": f_sigmoid,
            "interaction": f_interaction,
        }
        all_functions.update(self.polynomial_functions)
        all_functions.update(self.nonlinear_functions)
        return all_functions

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
        possible_functions = self._get_all_possible_functions()

        gen = SCMGenerator(
            all_functions=possible_functions,
            seed=0,
        )

        for function_to_use in possible_functions:
            print(f"Generating data for {function_to_use} function...")
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

    def get_adjacency_matrix(
        self, function_name: str = None, dataset_index: int = 0
    ) -> pd.DataFrame:
        """
        Returns the adjacency matrix representation of a DAG.

        Args:
            function_name (str, optional): The function name to get the adjacency matrix for.
                If None, returns the adjacency matrix for the first available function.
            dataset_index (int, optional): Index of the dataset to use. Default is 0.

        Returns:
            pd.DataFrame: Adjacency matrix where entry (i,j) is 1 if there is an edge from i to j.

        Raises:
            ValueError: If no data has been generated or the specified function_name is not available.
        """
        if not self.generated_dags:
            raise ValueError("No data has been generated. Call build() method first.")

        if function_name is None:
            function_name = list(self.generated_dags.keys())[0]

        if function_name not in self.generated_dags:
            available_functions = list(self.generated_dags.keys())
            raise ValueError(
                f"Function '{function_name}' not found. Available functions: {available_functions}"
            )

        if dataset_index >= len(self.generated_dags[function_name]):
            max_index = len(self.generated_dags[function_name]) - 1
            raise ValueError(
                f"Dataset index out of range. Max index for '{function_name}' is {max_index}"
            )

        dag = self.generated_dags[function_name][dataset_index]

        # Get the adjacency matrix
        adjacency_matrix = nx.to_pandas_adjacency(dag, nodelist=sorted(dag.nodes()))

        return adjacency_matrix

    def get_correlation_matrix(
        self, function_name: str = None, dataset_index: int = 0
    ) -> pd.DataFrame:
        """
        Computes the Pearson correlation matrix for a specified dataset.

        Args:
            function_name (str, optional): The function name to get the correlation matrix for.
                If None, returns the correlation matrix for the first available function.
            dataset_index (int, optional): Index of the dataset to use. Default is 0.

        Returns:
            pd.DataFrame: Correlation matrix where entry (i,j) represents the correlation between variables i and j.

        Raises:
            ValueError: If no data has been generated or the specified function_name is not available.
        """
        if not self.generated_observations:
            raise ValueError("No data has been generated. Call build() method first.")

        if function_name is None:
            function_name = list(self.generated_observations.keys())[0]

        if function_name not in self.generated_observations:
            available_functions = list(self.generated_observations.keys())
            raise ValueError(
                f"Function '{function_name}' not found. Available functions: {available_functions}"
            )

        if dataset_index >= len(self.generated_observations[function_name]):
            max_index = len(self.generated_observations[function_name]) - 1
            raise ValueError(
                f"Dataset index out of range. Max index for '{function_name}' is {max_index}"
            )

        # Get the dataset
        dataset = self.generated_observations[function_name][dataset_index]

        # Compute correlation matrix
        correlation_matrix = dataset.corr(method="pearson")

        return correlation_matrix

    def get_covariance_matrix(
        self, function_name: str = None, dataset_index: int = 0
    ) -> pd.DataFrame:
        """
        Computes the covariance matrix for a specified dataset.

        Args:
            function_name (str, optional): The function name to get the covariance matrix for.
                If None, returns the covariance matrix for the first available function.
            dataset_index (int, optional): Index of the dataset to use. Default is 0.

        Returns:
            pd.DataFrame: Covariance matrix where entry (i,j) represents the covariance between variables i and j.

        Raises:
            ValueError: If no data has been generated or the specified function_name is not available.
        """
        if not self.generated_observations:
            raise ValueError("No data has been generated. Call build() method first.")

        if function_name is None:
            function_name = list(self.generated_observations.keys())[0]

        if function_name not in self.generated_observations:
            available_functions = list(self.generated_observations.keys())
            raise ValueError(
                f"Function '{function_name}' not found. Available functions: {available_functions}"
            )

        if dataset_index >= len(self.generated_observations[function_name]):
            max_index = len(self.generated_observations[function_name]) - 1
            raise ValueError(
                f"Dataset index out of range. Max index for '{function_name}' is {max_index}"
            )

        # Get the dataset
        dataset = self.generated_observations[function_name][dataset_index]

        # Compute covariance matrix
        covariance_matrix = dataset.cov()

        return covariance_matrix

    def get_parcorr_matrix(
        self, function_name: str = None, dataset_index: int = 0
    ) -> pd.DataFrame:
        """
        Computes the partial correlation matrix for a specified dataset.

        Partial correlation measures the degree of association between two variables,
        with the effect of a set of controlling variables removed.

        Args:
            function_name (str, optional): The function name to get the partial correlation matrix for.
                If None, returns the partial correlation matrix for the first available function.
            dataset_index (int, optional): Index of the dataset to use. Default is 0.

        Returns:
            pd.DataFrame: Partial correlation matrix where entry (i,j) represents
                        the partial correlation between variables i and j.

        Raises:
            ValueError: If no data has been generated or the specified function_name is not available.
        """
        if not self.generated_observations:
            raise ValueError("No data has been generated. Call build() method first.")

        if function_name is None:
            function_name = list(self.generated_observations.keys())[0]

        if function_name not in self.generated_observations:
            available_functions = list(self.generated_observations.keys())
            raise ValueError(
                f"Function '{function_name}' not found. Available functions: {available_functions}"
            )

        if dataset_index >= len(self.generated_observations[function_name]):
            max_index = len(self.generated_observations[function_name]) - 1
            raise ValueError(
                f"Dataset index out of range. Max index for '{function_name}' is {max_index}"
            )

        # Get the dataset
        dataset = self.generated_observations[function_name][dataset_index]

        # Compute the inverse of the covariance matrix (precision matrix)
        import numpy as np
        from scipy import stats

        # Get the correlation matrix
        correlation_matrix = dataset.corr().values

        # Get the number of variables
        n_vars = correlation_matrix.shape[0]

        # Initialize the partial correlation matrix
        parcorr_matrix = np.zeros((n_vars, n_vars))

        # Compute the partial correlation
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                # Get the indices of control variables (all except i and j)
                control_idx = [k for k in range(n_vars) if k != i and k != j]

                # If there are control variables
                if control_idx:
                    # Get the data for variables i, j, and control variables
                    x = dataset.iloc[:, i].values
                    y = dataset.iloc[:, j].values
                    z = dataset.iloc[:, control_idx].values

                    # Residualize x with respect to control variables
                    beta_x = np.linalg.lstsq(z, x, rcond=None)[0]
                    res_x = x - z @ beta_x

                    # Residualize y with respect to control variables
                    beta_y = np.linalg.lstsq(z, y, rcond=None)[0]
                    res_y = y - z @ beta_y

                    # Compute correlation between residuals
                    parcorr_matrix[i, j] = parcorr_matrix[j, i] = np.corrcoef(
                        res_x, res_y
                    )[0, 1]
                else:
                    # If no control variables, partial correlation is just the correlation
                    parcorr_matrix[i, j] = parcorr_matrix[j, i] = correlation_matrix[
                        i, j
                    ]

        # Set diagonal to 1
        np.fill_diagonal(parcorr_matrix, 1.0)

        # Convert to DataFrame with variable names
        parcorr_df = pd.DataFrame(
            parcorr_matrix, index=dataset.columns, columns=dataset.columns
        )

        return parcorr_df

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
