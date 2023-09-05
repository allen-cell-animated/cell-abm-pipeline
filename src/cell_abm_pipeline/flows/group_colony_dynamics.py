"""
Workflow for grouping colony dynamics.

Working location structure:

.. code-block:: bash

    (name)
    ├── analysis
    │   ├── analysis.MEASURES
    │   │   └── (name)_(key).MEASURES.csv
    │   └── analysis.COLONIES
    │       └── (name)_(key).COLONIES.csv
    └── groups
        └── groups.COLONIES
            ├── (name).feature_distributions.(feature).json
            ├── (name).feature_temporal.(key).(feature).json
            ├── (name).neighbor_positions.(key).(seed).(tick).csv
            └── (name).neighbor_positions.(key).(seed).(tick).(feature).csv

Different groups use inputs from the **analysis/analysis.COLONIES** and
**analysis/analysis.MEASURES** directories.
Grouped data is saved to the **groups/groups.COLONIES** directory.

Different groups can be visualized using the corresponding plotting workflow or
loaded into alternative tools.
"""

import ast
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd
from io_collection.keys import make_key
from io_collection.load import load_dataframe
from io_collection.save import save_dataframe, save_json
from prefect import flow
from prefect.tasks import task_input_hash

from cell_abm_pipeline.tasks import calculate_data_bins, check_data_bounds

OPTIONS = {
    "cache_result_in_memory": False,
    "cache_key_fn": task_input_hash,
    "cache_expiration": timedelta(hours=12),
}

GROUPS: list[str] = [
    "feature_distributions",
    "feature_temporal",
    "neighbor_positions",
]

DISTRIBUTION_FEATURES: list[str] = [
    "degree",
    "eccentricity",
    "degree_centrality",
    "closeness_centrality",
    "betweenness_centrality",
]

TEMPORAL_FEATURES: list[str] = [
    "degree",
    "eccentricity",
    "degree_centrality",
    "closeness_centrality",
    "betweenness_centrality",
    "radius",
    "diameter",
]

POSITION_FEATURES: list[str] = [
    "depth",
    "group",
]

BOUNDS: dict[str, list] = {
    "degree": [-1, 30],
    "eccentricity": [-1, 15],
    "degree_centrality": [-0.1, 1],
    "closeness_centrality": [-0.1, 1],
    "betweenness_centrality": [-0.1, 1],
}

BANDWIDTH: dict[str, float] = {
    "degree": 1,
    "eccentricity": 1,
    "degree_centrality": 0.05,
    "closeness_centrality": 0.05,
    "betweenness_centrality": 0.05,
}


@dataclass
class ParametersConfigFeatureDistributions:
    """Parameter configuration for group colony dynamics subflow - feature distributions."""

    features: list[str] = field(default_factory=lambda: DISTRIBUTION_FEATURES)
    """List of colony features."""

    bounds: dict[str, list] = field(default_factory=lambda: BOUNDS)
    """Bounds for feature distributions."""

    bandwidth: dict[str, float] = field(default_factory=lambda: BANDWIDTH)
    """Bandwidths for feature distributions."""


@dataclass
class ParametersConfigFeatureTemporal:
    """Parameter configuration for group colony dynamics subflow - feature temporal."""

    features: list[str] = field(default_factory=lambda: TEMPORAL_FEATURES)
    """List of temporal features."""


@dataclass
class ParametersConfigNeighborPositions:
    """Parameter configuration for group colony dynamics subflow - neighbor positions."""

    features: list[str] = field(default_factory=lambda: POSITION_FEATURES)
    """List of position features."""

    seed: int = 0
    """Simulation seed to use for grouping neighbor positions."""

    ticks: list[int] = field(default_factory=lambda: [0])
    """Simulation ticks to use for grouping neighbor positions."""


@dataclass
class ParametersConfig:
    """Parameter configuration for group colony dynamics flow."""

    groups: list[str] = field(default_factory=lambda: GROUPS)
    """List of colony dynamics groups."""

    feature_distributions: ParametersConfigFeatureDistributions = (
        ParametersConfigFeatureDistributions()
    )
    """Parameters for group feature distributions subflow."""

    feature_temporal: ParametersConfigFeatureTemporal = ParametersConfigFeatureTemporal()
    """Parameters for group feature temporal subflow."""

    neighbor_positions: ParametersConfigNeighborPositions = ParametersConfigNeighborPositions()
    """Parameters for group neighbor positions subflow."""


@dataclass
class ContextConfig:
    """Context configuration for group colony dynamics flow."""

    working_location: str
    """Location for input and output files (local path or S3 bucket)."""


@dataclass
class SeriesConfig:
    """Series configuration for group colony dynamics flow."""

    name: str
    """Name of the simulation series."""

    conditions: list[dict]
    """List of series condition dictionaries (must include unique condition "key")."""


@flow(name="group-colony-dynamics")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    """
    Main group colony dynamics flow.

    Calls the following subflows, if the group is specified:

    - :py:func:`run_flow_group_feature_distributions`
    - :py:func:`run_flow_group_feature_temporal`
    - :py:func:`run_flow_group_neighbor_positions`
    """

    if "feature_distributions" in parameters.groups:
        run_flow_group_feature_distributions(context, series, parameters.feature_distributions)

    if "feature_temporal" in parameters.groups:
        run_flow_group_feature_temporal(context, series, parameters.feature_temporal)

    if "neighbor_positions" in parameters.groups:
        run_flow_group_neighbor_positions(context, series, parameters.neighbor_positions)


@flow(name="group-colony-dynamics_group-feature-distributions")
def run_flow_group_feature_distributions(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigFeatureDistributions
) -> None:
    """Group colony dynamics subflow for feature distributions."""

    analysis_key = make_key(series.name, "analysis", "analysis.MEASURES")
    group_key = make_key(series.name, "groups", "groups.COLONIES")
    keys = [condition["key"] for condition in series.conditions]

    distribution_bins: dict[str, dict] = {feature: {} for feature in parameters.features}
    distribution_means: dict[str, dict] = {feature: {} for feature in parameters.features}
    distribution_stdevs: dict[str, dict] = {feature: {} for feature in parameters.features}

    for key in keys:
        # Load dataframe.
        dataframe_key = make_key(analysis_key, f"{series.name}_{key}.MEASURES.csv")
        data = load_dataframe.with_options(**OPTIONS)(context.working_location, dataframe_key)

        for feature in parameters.features:
            values = data[feature.upper()].values

            bounds = (parameters.bounds[feature][0], parameters.bounds[feature][1])
            bandwidth = parameters.bandwidth[feature]

            check_data_bounds(values, bounds, f"[ {key} ] feature [ {feature} ]")

            distribution_means[feature][key] = np.mean(values)
            distribution_stdevs[feature][key] = np.std(values, ddof=1)
            distribution_bins[feature][key] = calculate_data_bins(values, bounds, bandwidth)

    for feature, distribution in distribution_bins.items():
        distribution["*"] = {
            "bandwidth": parameters.bandwidth[feature],
            "means": distribution_means[feature],
            "stdevs": distribution_stdevs[feature],
        }

        save_json(
            context.working_location,
            make_key(group_key, f"{series.name}.feature_distributions.{feature.upper()}.json"),
            distribution,
        )


@flow(name="group-colony-dynamics_group-feature-temporal")
def run_flow_group_feature_temporal(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigFeatureTemporal
) -> None:
    """Group colony dynamics subflow for temporal features."""

    analysis_key = make_key(series.name, "analysis", "analysis.MEASURES")
    group_key = make_key(series.name, "groups", "groups.COLONIES")
    keys = [condition["key"] for condition in series.conditions]

    for key in keys:
        # Load dataframe.
        dataframe_key = make_key(analysis_key, f"{series.name}_{key}.MEASURES.csv")
        data = load_dataframe.with_options(**OPTIONS)(context.working_location, dataframe_key)

        for feature in parameters.features:
            if feature == "radius":
                values = data.groupby(["SEED", "time"])["ECCENTRICITY"].min().groupby(["time"])
            elif feature == "diameter":
                values = data.groupby(["SEED", "time"])["ECCENTRICITY"].max().groupby(["time"])
            else:
                values = data.groupby(["SEED", "time"])[feature.upper()].mean().groupby(["time"])

            temporal = {
                "time": list(values.groups.keys()),
                "mean": [v if not np.isnan(v) else "nan" for v in values.mean()],
                "std": [v if not np.isnan(v) else "nan" for v in values.std(ddof=1)],
                "min": [v if not np.isnan(v) else "nan" for v in values.min()],
                "max": [v if not np.isnan(v) else "nan" for v in values.max()],
            }

            save_json(
                context.working_location,
                make_key(group_key, f"{series.name}.feature_temporal.{key}.{feature.upper()}.json"),
                temporal,
            )


@flow(name="group-colony-dynamics_group-neighbor-positions")
def run_flow_group_neighbor_positions(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigNeighborPositions
) -> None:
    """Group colony dynamics subflow for neighbor positions."""

    analysis_key = make_key(series.name, "analysis", "analysis.COLONIES")
    group_key = make_key(series.name, "groups", "groups.COLONIES")
    keys = [condition["key"] for condition in series.conditions]

    for key in keys:
        dataframe_key = make_key(analysis_key, f"{series.name}_{key}.COLONIES.csv")
        data = load_dataframe.with_options(**OPTIONS)(
            context.working_location, dataframe_key, converters={"NEIGHBORS": ast.literal_eval}
        )
        groups = data[data["SEED"] == parameters.seed].groupby("TICK")

        for tick in parameters.ticks:
            if tick not in groups.groups:
                continue

            group = groups.get_group(tick)

            # Save edge data.
            edges = set()
            for item in group[["ID", "NEIGHBORS"]].to_dict("records"):
                edges.update(
                    {tuple(sorted([item["ID"], neighbor])) for neighbor in item["NEIGHBORS"]}
                )

            edge_key = f"{key}.{parameters.seed:04d}.{tick:06d}"
            save_dataframe(
                context.working_location,
                make_key(group_key, f"{series.name}.neighbor_positions.{edge_key}.csv"),
                pd.DataFrame(list(edges), columns=["id1", "id2"]),
                index=False,
            )

            # Save node data for each feature.
            for feature in parameters.features:
                nodes = group[["ID", "cx", "cy", "cz", feature.upper()]].rename(
                    columns={"ID": "id", "cx": "x", "cy": "y", "cz": "z", feature.upper(): "v"}
                )

                node_key = f"{key}.{parameters.seed:04d}.{tick:06d}.{feature.upper()}"
                save_dataframe(
                    context.working_location,
                    make_key(group_key, f"{series.name}.neighbor_positions.{node_key}.csv"),
                    nodes,
                    index=False,
                )
