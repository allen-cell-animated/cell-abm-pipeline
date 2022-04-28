import numpy as np
import pandas as pd
import networkx as nx

from cell_abm_pipeline.utilities.load import load_pickle
from cell_abm_pipeline.utilities.save import save_dataframe
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key


class AnalyzeMeasures:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "analysis", "NETWORKS", True),
            "output": make_folder_key(context.name, "analysis", "MEASURES", True),
        }
        self.files = {
            "input": make_file_key(context.name, ["NETWORKS", "pkl"], "%s", "%04d"),
            "output": make_file_key(context.name, ["MEASURES", "csv"], "%s", ""),
        }

    def run(self):
        for key in self.context.keys:
            self.analyze_measures(key)

    def analyze_measures(self, key):
        all_data = []

        for seed in self.context.seeds:
            file_key = self.folders["input"] + self.files["input"] % (key, seed)
            data = load_pickle(self.context.working, file_key)
            all_data = all_data + [(seed, key, value) for key, value in data.items()]

        output_key = self.folders["output"] + self.files["output"] % (key)
        output_df = self.calculate_graph_measures(all_data)
        save_dataframe(self.context.working, output_key, output_df, index=False)

    @staticmethod
    def calculate_graph_measures(graphs):
        all_measures = []

        for seed, tick, graph in graphs:
            degrees, degree_mean, degree_std = AnalyzeMeasures.get_network_degrees(graph)
            radius, diameter, eccentricity, path = AnalyzeMeasures.get_network_distances(graph)
            degree, closeness, betweenness = AnalyzeMeasures.get_network_centralities(graph)

            measure = {
                "SEED": seed,
                "TICK": tick,
                "DEGREES": degrees,
                "DEGREE_MEAN": degree_mean,
                "DEGREE_STD": degree_std,
                "RADIUS": radius,
                "DIAMETER": diameter,
                "ECCENTRICITY": eccentricity,
                "SHORTEST_PATH": path,
                "DEGREE_CENTRALITY": degree,
                "CLOSENESS_CENTRALITY": closeness,
                "BETWEENNESS_CENTRALITY": betweenness,
            }

            all_measures.append(measure)

        return pd.DataFrame(all_measures)

    @staticmethod
    def get_network_degrees(graph):
        """Gets degrees of network."""
        if nx.is_empty(graph):
            return np.nan, np.nan, np.nan

        degrees = sorted([d for n, d in graph.degree()], reverse=True)
        degree_mean = np.mean(degrees)
        degree_std = np.std(degrees, ddof=1)
        return degrees, degree_mean, degree_std

    @staticmethod
    def get_network_distances(graph):
        """Calculates network distance measures."""
        if not nx.is_connected(graph):
            subgraphs = [graph.subgraph(component) for component in nx.connected_components(graph)]

            radii = [nx.radius(subgraph) for subgraph in subgraphs]
            radius = np.mean(radii)

            diameters = [nx.diameter(subgraph) for subgraph in subgraphs]
            diameter = np.mean(diameters)

            eccentricities = [nx.eccentricity(subgraph) for subgraph in subgraphs]
            eccentricity = np.mean([np.mean(list(ecc.values())) for ecc in eccentricities])

            paths = [nx.average_shortest_path_length(subgraph) for subgraph in subgraphs]
            path = np.mean(paths)
        else:
            radius = nx.radius(graph)

            diameter = nx.diameter(graph)

            ecc = nx.eccentricity(graph)
            eccentricity = np.mean(list(ecc.values()))

            path = nx.average_shortest_path_length(graph)

        return radius, diameter, eccentricity, path

    @staticmethod
    def get_network_centralities(graph):
        """Calculates network centrality measures."""
        degree = nx.degree_centrality(graph)
        closeness = nx.closeness_centrality(graph)
        betweenness = nx.betweenness_centrality(graph)

        avg_deg = np.mean(list(degree.values()))
        avg_clos = np.mean(list(closeness.values()))
        avg_betw = np.mean(list(betweenness.values()))

        return avg_deg, avg_clos, avg_betw
