"""
Workflow for analyzing colony dynamics.
"""

import ast
from dataclasses import dataclass

import pandas as pd
from abm_colony_collection import (
    calculate_centrality_measures,
    calculate_cluster_distances,
    calculate_cluster_sizes,
    calculate_degree_measures,
    calculate_distance_measures,
    convert_to_network,
)
from arcade_collection.output import convert_model_units
from io_collection.keys import check_key, make_key
from io_collection.load import load_dataframe, load_pickle
from io_collection.save import save_dataframe, save_pickle
from prefect import flow


@dataclass
class ParametersConfig:
    ds: float = 1.0

    dt: float = 1.0


@dataclass
class ContextConfig:
    working_location: str


@dataclass
class SeriesConfig:
    name: str

    seeds: list[int]

    conditions: list[dict]


@flow(name="analyze-colony-dynamics")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    # Process neighbor connections to generate graph objects where nodes
    # represent cells and edges represent cells that share borders. If the
    # network already exists for a given key and seed, that key and seed are
    # skipped.
    run_flow_generate_networks(context, series, parameters)

    # Perform graph analysis on neighbor connections. If the analysis file
    # already exists for a given key, that key is skipped.
    run_flow_analyze_measures(context, series, parameters)

    # Perform cluster analysis on neighbor connections. If the analysis file
    # already exists for a given key, that key is skipped.
    run_flow_analyze_clusters(context, series, parameters)


@flow(name="analyze-colony-dynamics_generate-networks")
def run_flow_generate_networks(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    neighbors_path_key = make_key(series.name, "analysis", "analysis.NEIGHBORS")
    networks_path_key = make_key(series.name, "analysis", "analysis.NETWORKS")
    keys = [condition["key"] for condition in series.conditions]

    for key in keys:
        networks_key = make_key(networks_path_key, f"{series.name}_{key}.NETWORKS.pkl")

        if check_key(context.working_location, networks_key):
            continue

        all_networks = {}

        for seed in series.seeds:
            neighbors_key = make_key(
                neighbors_path_key, f"{series.name}_{key}_{seed:04d}.NEIGHBORS.csv"
            )
            neighbors = load_dataframe(
                context.working_location, neighbors_key, converters={"NEIGHBORS": ast.literal_eval}
            )
            networks = convert_to_network(neighbors)
            all_networks.update({(seed, tick): network for tick, network in networks.items()})

        save_pickle(context.working_location, networks_key, all_networks)


@flow(name="analyze-colony-dynamics_analyze-measures")
def run_flow_analyze_measures(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    networks_path_key = make_key(series.name, "analysis", "analysis.NETWORKS")
    measures_path_key = make_key(series.name, "analysis", "analysis.MEASURES")
    keys = [condition["key"] for condition in series.conditions]

    for key in keys:
        measures_key = make_key(measures_path_key, f"{series.name}_{key}.MEASURES.csv")

        if check_key(context.working_location, measures_key):
            continue

        networks_key = make_key(networks_path_key, f"{series.name}_{key}.NETWORKS.pkl")
        networks = load_pickle(context.working_location, networks_key)

        degree_measures = calculate_degree_measures(networks)
        distance_measures = calculate_distance_measures(networks)
        centrality_measures = calculate_centrality_measures(networks)

        measures = degree_measures.merge(distance_measures, on=["SEED", "TICK"])
        measures = measures.merge(centrality_measures, on=["SEED", "TICK"])
        convert_model_units(measures, parameters.ds, parameters.dt)

        save_dataframe(context.working_location, measures_key, measures, index=False)


@flow(name="analyze-colony-dynamics_analyze-clusters")
def run_flow_analyze_clusters(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    neighbors_path_key = make_key(series.name, "analysis", "analysis.NEIGHBORS")
    clusters_path_key = make_key(series.name, "analysis", "analysis.CLUSTERS")
    keys = [condition["key"] for condition in series.conditions]

    for key in keys:
        clusters_key = make_key(clusters_path_key, f"{series.name}_{key}.CLUSTERS.csv")

        if check_key(context.working_location, clusters_key):
            continue

        all_neighbors = []

        for seed in series.seeds:
            neighbors_key = make_key(
                neighbors_path_key, f"{series.name}_{key}_{seed:04d}.NEIGHBORS.csv"
            )
            neighbors = load_dataframe(
                context.working_location, neighbors_key, converters={"NEIGHBORS": ast.literal_eval}
            )
            all_neighbors.append(neighbors)

        neighbors_data = pd.concat(all_neighbors)
        convert_model_units(neighbors_data, parameters.ds, parameters.dt)

        cluster_sizes = calculate_cluster_sizes(neighbors_data)
        cluster_distances = calculate_cluster_distances(neighbors_data)

        clusters = cluster_sizes.merge(cluster_distances, on=["SEED", "TICK"])
        convert_model_units(clusters, parameters.ds, parameters.dt)

        save_dataframe(context.working_location, clusters_key, clusters, index=False)
