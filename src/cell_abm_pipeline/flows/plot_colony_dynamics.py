"""
Workflow for plotting colony dynamics.
"""

import ast
from dataclasses import dataclass, field

from io_collection.keys import make_key
from io_collection.load import load_dataframe
from io_collection.save import save_figure
from prefect import flow

from cell_abm_pipeline.tasks.clusters import (
    plot_cluster_counts,
    plot_cluster_fractions,
    plot_cluster_trajectory,
)
from cell_abm_pipeline.tasks.measures import (
    plot_degree_distribution,
    plot_degree_trajectory,
    plot_graph_centralities,
    plot_graph_distances,
)

PLOTS_MEASURES = [
    "degree_distribution",
    "degree_means",
    "degree_stds",
    "graph_centralities",
    "graph_distances",
]

PLOTS_CLUSTERS = [
    "cluster_counts",
    "cluster_fractions",
    "interdistance_mean",
    "interdistance_std",
    "intradistance_mean",
    "intradistance_std",
    "size_means",
    "size_stds",
]

PLOTS = PLOTS_MEASURES + PLOTS_CLUSTERS


@dataclass
class ParametersConfig:
    """Parameter configuration for plot colony dynamics flow."""

    plots: list[str] = field(default_factory=lambda: PLOTS)


@dataclass
class ContextConfig:
    """Context configuration for plot colony dynamics flow."""

    working_location: str


@dataclass
class SeriesConfig:
    """Series configuration for plot colony dynamics flow."""

    name: str

    conditions: list[dict]


@flow(name="plot-colony-dynamics")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    # Make plots for graph analysis on neighbor connections.
    if any(plot in parameters.plots for plot in PLOTS_MEASURES):
        run_flow_plot_measures(context, series, parameters)

    # Make plots for cluster analysis on neighbor connections.
    if any(plot in parameters.plots for plot in PLOTS_CLUSTERS):
        run_flow_plot_clusters(context, series, parameters)


@flow(name="plot-colony-dynamics_plot-measures")
def run_flow_plot_measures(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    analysis_key = make_key(series.name, "analysis", "analysis.MEASURES")
    plot_key = make_key(series.name, "plots", "plots.MEASURES")
    keys = [condition["key"] for condition in series.conditions]

    all_measures = {}

    for key in keys:
        measures_key = make_key(analysis_key, f"{series.name}_{key}.MEASURES.csv")
        measures = load_dataframe(
            context.working_location, measures_key, converters={"DEGREES": ast.literal_eval}
        )
        all_measures[key] = measures

    if "degree_distribution" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_degree_distribution.MEASURES.png"),
            plot_degree_distribution(keys, all_measures),
        )

    if "degree_means" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_degree_means.MEASURES.png"),
            plot_degree_trajectory(keys, all_measures, "DEGREE_MEAN"),
        )

    if "degree_stds" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_degree_stds.MEASURES.png"),
            plot_degree_trajectory(keys, all_measures, "DEGREE_STD"),
        )

    if "graph_centralities" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_graph_centralities.MEASURES.png"),
            plot_graph_centralities(keys, all_measures),
        )

    if "graph_distances" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_graph_distances.MEASURES.png"),
            plot_graph_distances(keys, all_measures),
        )


@flow(name="plot-colony-dynamics_plot-clusters")
def run_flow_plot_clusters(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    analysis_key = make_key(series.name, "analysis", "analysis.CLUSTERS")
    plot_key = make_key(series.name, "plots", "plots.CLUSTERS")
    keys = [condition["key"] for condition in series.conditions]

    all_clusters = {}

    for key in keys:
        clusters_key = make_key(analysis_key, f"{series.name}_{key}.CLUSTERS.csv")
        clusters = load_dataframe(
            context.working_location,
            clusters_key,
            converters={
                "INTRA_DISTANCE_MEAN": ast.literal_eval,
                "INTRA_DISTANCE_STD": ast.literal_eval,
            },
        )
        all_clusters[key] = clusters

    if "cluster_counts" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_cluster_counts.CLUSTERS.png"),
            plot_cluster_counts(keys, all_clusters),
        )

    if "cluster_fractions" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_cluster_fractions.CLUSTERS.png"),
            plot_cluster_fractions(keys, all_clusters),
        )

    if "interdistance_mean" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_interdistance_mean.CLUSTERS.png"),
            plot_cluster_trajectory(keys, all_clusters, "INTER_DISTANCE_MEAN"),
        )

    if "interdistance_std" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_interdistance_std.CLUSTERS.png"),
            plot_cluster_trajectory(keys, all_clusters, "INTER_DISTANCE_STD"),
        )

    if "intradistance_mean" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_intradistance_mean.CLUSTERS.png"),
            plot_cluster_trajectory(keys, all_clusters, "INTRA_DISTANCE_MEAN"),
        )

    if "intradistance_std" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_intradistance_std.CLUSTERS.png"),
            plot_cluster_trajectory(keys, all_clusters, "INTRA_DISTANCE_STD"),
        )

    if "size_means" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_size_means.CLUSTERS.png"),
            plot_cluster_trajectory(keys, all_clusters, "CLUSTER_SIZE_MEAN"),
        )

    if "size_stds" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_size_stds.CLUSTERS.png"),
            plot_cluster_trajectory(keys, all_clusters, "CLUSTER_SIZE_STD"),
        )
