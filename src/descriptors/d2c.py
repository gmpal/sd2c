from multiprocessing import Pool
import pandas as pd
import numpy as np
import itertools
from scipy.stats import kurtosis, skew

from .utils import coeff, HOC, update_dictionary_quantiles
from .estimators import (
    MarkovBlanketEstimator,
    MutualInformationEstimator,
)


class D2C:
    """ """

    def __init__(
        self,
        dags,
        observations,
        MB_size=5,
        n_variables=3,
        maxlags=3,
        mutual_information_proxy="Ridge",
        proxy_params=None,
        verbose=False,
        seed=42,
        n_jobs=1,
    ) -> None:

        self.DAGs = dags
        self.observations = observations
        self.mutual_information_proxy = mutual_information_proxy
        self.proxy_params = proxy_params
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.seed = seed

        self.markov_blanket_estimator = MarkovBlanketEstimator(
            size=min(MB_size, n_variables - 2), n_variables=n_variables, maxlags=maxlags
        )

        self.mutual_information_estimator = MutualInformationEstimator(
            proxy=mutual_information_proxy,
            proxy_params=proxy_params,
            k=None,
        )

        np.random.seed(seed)

    def initialize(self) -> None:
        """
        Initialize the D2C object by computing descriptors in parallel for all observations.
        """
        if self.n_jobs == 1:
            results = [
                self.compute_descriptors(dag_idx, dag)
                for dag_idx, dag in enumerate(self.DAGs)
            ]
        else:
            args = [(dag_idx, dag) for dag_idx, dag in enumerate(self.DAGs)]
            with Pool(processes=self.n_jobs) as pool:
                results = pool.starmap(self.compute_descriptors, args)

        # merge lists into a single list
        results = [item for sublist in results for item in sublist]
        self.descriptors_df = results

    def compute_descriptors(self, dag_idx, dag):
        """
        Compute descriptors for a given couple of nodes in a directed acyclic graph (DAG).

        Args:
            dag_idx (int): The index of the DAG.
            ca (int): The index of the cause node.
            ef (int): The index of the effect node.
            label (bool): The label indicating whether the edge between the cause and effect nodes is causal.

        Returns:
            dict: A dictionary containing the computed descriptors.

        """
        all_possible_links = list(itertools.permutations(list(dag.nodes()), 2))
        collected_values = []
        for ca_name, ef_name in all_possible_links:
            label = (ca_name, ef_name) in list(dag.edges())

            observations_df = self.observations[dag_idx].copy()

            # normalization is not performed to preserve causal structures !
            # observations_df = (
            #     self.observations[dag_idx] - self.observations[dag_idx].mean()
            # ) / self.observations[dag_idx].std()

            # # fillna
            # observations_df = observations_df.fillna(0)  # constant columns

            ca = list(observations_df.columns).index(ca_name)
            ef = list(observations_df.columns).index(ef_name)

            obs = observations_df.to_numpy()

            MBca = self.markov_blanket_estimator.estimate(obs, node=ca)
            MBef = self.markov_blanket_estimator.estimate(obs, node=ef)

            common_causes = list(set(MBca).intersection(MBef))
            mbca_mbef = list(set([(i, j) for i in MBca for j in MBef]))
            mbca_mbca = list(set([(i, j) for i in MBca for j in MBca]))
            mbef_mbef = list(set([(i, j) for i in MBef for j in MBef]))

            CMI = self.mutual_information_estimator.estimate_original

            values = {}
            values_qt = {}
            values["graph_id"] = dag_idx
            values["edge_source"] = ca_name
            values["edge_dest"] = ef_name
            values["is_causal"] = label

            # b: ef = b * (ca + mbef)
            values["coeff_cause"] = coeff(obs[:, ef], obs[:, ca], obs[:, MBef])

            # b: ca = b * (ef + mbca)
            values["coeff_eff"] = coeff(obs[:, ca], obs[:, ef], obs[:, MBca])

            values["HOC_3_1"] = HOC(obs[:, ca], obs[:, ef], 3, 1)
            values["HOC_1_2"] = HOC(obs[:, ca], obs[:, ef], 1, 2)
            values["HOC_2_1"] = HOC(obs[:, ca], obs[:, ef], 2, 1)
            values["HOC_1_3"] = HOC(obs[:, ca], obs[:, ef], 1, 3)

            values["kurtosis_ca"] = kurtosis(obs[:, ca])
            values["kurtosis_ef"] = kurtosis(obs[:, ef])
            values["skewness_ca"] = skew(obs[:, ca])
            values["skewness_ef"] = skew(obs[:, ef])

            # I(cause; effect | common_causes)
            values["com_cau"] = CMI(obs, ef, ca, common_causes)

            # I(cause; effect)
            values["cau_eff"] = CMI(obs, ef, ca)

            # I(effect; cause)
            values["eff_cau"] = CMI(obs, ca, ef)

            # I(effect; cause | MBeffect)
            values["eff_cau_mbeff"] = CMI(obs, ca, ef, MBef)

            # I(cause; effect | MBcause)
            values["cau_eff_mbcau"] = CMI(obs, ef, ca, MBca)

            # I(mca ; mef | cause) for (mca,mef) in mbca_mbef_couples
            values_qt["mca_mef_cau"] = [CMI(obs, i, j, ca) for i, j in mbca_mbef]

            # I(mca ; mef| effect) for (mca,mef) in mbca_mbef_couples
            values_qt["mca_mef_eff"] = [CMI(obs, i, j, ef) for i, j in mbca_mbef]

            # I(cause; m | effect) for m in MBef
            values_qt["cau_m_eff"] = [CMI(obs, ca, m, ef) for m in MBef]

            # I(effect; m | cause) for m in MBca
            values_qt["eff_m_cau"] = [CMI(obs, ef, m, ca) for m in MBca]

            values_qt["m_cau"] = [CMI(obs, ca, m) for m in MBef]

            # I(effect; cause | arrays_m_plus_MBca)
            values_qt["eff_cau_mbcau_plus"] = [
                CMI(obs, ca, ef, np.r_[[m], MBca]) for m in MBef
            ]

            # I(cause; effect | arrays_m_plus_MBef)
            values_qt["cau_eff_mbeff_plus"] = [
                CMI(obs, ef, ca, np.r_[[m], MBef]) for m in MBca
            ]

            # I(m; effect) for m in MBca
            values_qt["m_eff"] = [CMI(obs, ef, m) for m in MBca]

            # I(mca ; mca| cause) - I(mca ; mca) for (mca,mca) in mbca_couples
            values_qt["mca_mca_cau"] = [
                CMI(obs, i, j, ca) - CMI(obs, i, j) for i, j in mbca_mbca
            ]

            # I(mbe ; mbe| effect) - I(mbe ; mbe) for (mbe,mbe) in mbef_couples
            values_qt["mbe_mbe_eff"] = [
                CMI(obs, i, j, ef) - CMI(obs, i, j) for i, j in mbef_mbef
            ]

            update_dictionary_quantiles(values, values_qt)

            collected_values.append(values)

        return collected_values
