import pickle
import pandas as pd
import itertools


class DataLoader:
    def __init__(self):
        self.observations = None
        self.dags = None

    def _flatten(self, dict_of_lists_of_elems):
        list_of_elems = []
        for function in sorted(dict_of_lists_of_elems.keys()):
            list_of_elems.extend(dict_of_lists_of_elems[function])
        return list_of_elems

    def from_pickle(self, data_path):

        with open(data_path, "rb") as f:
            loaded_observations, loaded_dags = pickle.load(f)
        self.observations = self._flatten(loaded_observations)
        self.dags = self._flatten(loaded_dags)

    def from_builder(self, builder):
        loaded_observations = builder.get_generated_observations()
        loaded_dags = builder.get_generated_dags()
        self.observations = self._flatten(loaded_observations)
        self.dags = self._flatten(loaded_dags)

    def get_observations(self):

        return self.observations

    def get_dags(self):

        return self.dags

    def get_true_causal_dfs(self):

        causal_dataframes = {}
        for dag_idx, dag in enumerate(self.dags):
            couples = list(itertools.permutations(list(dag.nodes()), 2))
            index = pd.MultiIndex.from_tuples(couples, names=["From", "To"])
            causal_dataframe = pd.DataFrame(
                index=index,
                columns=["effect", "p_value", "probability", "is_causal"],
                dtype=int,
            )

            causal_dataframe["effect"] = None
            causal_dataframe["p_value"] = None
            causal_dataframe["probability"] = None
            causal_dataframe["is_causal"] = 0

            for edge in dag.edges:
                causal_dataframe.loc[edge, "is_causal"] = 1

            causal_dataframe.reset_index(inplace=True)

            causal_dataframes[dag_idx] = causal_dataframe

        return causal_dataframes
