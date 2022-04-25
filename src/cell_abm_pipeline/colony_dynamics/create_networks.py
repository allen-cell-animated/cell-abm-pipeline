import pandas as pd
import networkx as nx

from cell_abm_pipeline.utilities.load import load_dataframe
from cell_abm_pipeline.utilities.save import save_pickle
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key


class CreateNetworks:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "analysis", "NEIGHBORS", False),
            "output": make_folder_key(context.name, "analysis", "NETWORKS", True),
        }
        self.files = {
            "input": make_file_key(context.name, ["NEIGHBORS", "csv", "xz"], "", "%04d"),
            "output": make_file_key(context.name, ["NETWORKS", "pkl"], "%s", "%04d"),
        }

    def run(self):
        self.create_networks()

    def create_networks(self):
        all_data = []

        for seed in self.context.seeds:
            file_key = self.folders["input"] + self.files["input"] % (seed)
            data = load_dataframe(self.context.working, file_key)
            data = data[data.KEY.isin(self.context.keys)]
            all_data.append(data)

        for (key, seed), key_group in pd.concat(all_data).groupby(["KEY", "SEED"]):
            output_key = self.folders["output"] + self.files["output"] % (key, seed)
            output = self.convert_to_network(key_group)
            save_pickle(self.context.working, output_key, output)

    @staticmethod
    def convert_to_network(neighbors):
        """Converts lists of cell ids and neighbors to networks."""
        all_networks = {}

        for tick, tick_group in neighbors.groupby("TICK"):
            edges = list(tick_group[["ID", "NEIGHBOR"]].to_records(index=False))
            G = nx.Graph()

            singles = [from_node for from_node, to_node in edges if to_node == 0]
            clusters = [(from_node, to_node) for from_node, to_node in edges if to_node != 0]

            G.add_nodes_from(singles)
            G.add_edges_from(clusters)

            all_networks[f"{tick:06d}"] = G

        return all_networks
