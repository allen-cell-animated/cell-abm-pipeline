"""
Workflow for grouping basic metrics.

Working location structure:

.. code-block:: bash

    (name)
    ├── groups
    │   └── groups.BASIC
    │       ├── (name).metrics_bins.(key).(seed).(tick).csv
    │       ├── (name).metrics_bins.(key).(seed).(tick).csv
    │       ├── ...
    │       ├── (name).metrics_bins.(key).(seed).(tick).csv
    │       ├── (name).metrics_distributions.(metric).json
    │       ├── (name).metrics_distributions.(metric).json
    │       ├── ...
    │       ├── (name).metrics_distributions.(metric).json
    │       ├── (name).metrics_individuals.(key).(seed).(metric).json
    │       ├── (name).metrics_individuals.(key).(seed).(metric).json
    │       ├── ...
    │       ├── (name).metrics_individuals.(key).(seed).(metric).json
    │       ├── (name).metrics_spatial.(key).(seed).(tick).(metric).csv
    │       ├── (name).metrics_spatial.(key).(seed).(tick).(metric).csv
    │       ├── ...
    │       ├── (name).metrics_spatial.(key).(seed).(tick).(metric).csv
    │       ├── (name).metrics_temporal.(key).(metric).json
    │       ├── (name).metrics_temporal.(key).(metric).json
    │       ├── ...
    │       ├── (name).metrics_temporal.(key).(metric).json
    │       └── (name).population_counts.csv
    └── results
        ├── (name)_(key)_(seed).csv
        ├── (name)_(key)_(seed).csv
        ├── ...
        └── (name)_(key)_(seed).csv

Different groups use inputs from the **results** directory.
Grouped data is saved to the **groups/groups.BASIC** directory.

Different groups can be visualized using the corresponding plotting workflow or
loaded into alternative tools.
"""

import ast
from dataclasses import dataclass, field
from itertools import groupby

import numpy as np
import pandas as pd
from arcade_collection.output import convert_model_units
from io_collection.keys import check_key, make_key
from io_collection.load import load_dataframe
from io_collection.save import save_dataframe, save_json
from prefect import flow, get_run_logger

from cell_abm_pipeline.tasks import bin_to_hex, calculate_category_durations, calculate_data_bins

GROUPS: list[str] = [
    "metrics_bins",
    "metrics_distributions",
    "metrics_individuals",
    "metrics_spatial",
    "metrics_temporal",
    "population_counts",
]

CELL_PHASES: list[str] = [
    "PROLIFERATIVE_G1",
    "PROLIFERATIVE_S",
    "PROLIFERATIVE_G2",
    "PROLIFERATIVE_M",
    "APOPTOTIC_EARLY",
    "APOPTOTIC_LATE",
]

BOUNDS: dict[str, list] = {
    "volume.DEFAULT": [0, 5000],
    "volume.NUCLEUS": [0, 1500],
    "height.DEFAULT": [0, 20],
    "height.NUCLEUS": [0, 20],
    "phase.PROLIFERATIVE_G1": [0, 5],
    "phase.PROLIFERATIVE_S": [0, 20],
    "phase.PROLIFERATIVE_G2": [0, 18],
    "phase.PROLIFERATIVE_M": [0, 2],
    "phase.APOPTOTIC_EARLY": [0, 6],
    "phase.APOPTOTIC_LATE": [0, 12],
}

BANDWIDTH: dict[str, float] = {
    "volume.DEFAULT": 100,
    "volume.NUCLEUS": 50,
    "height.DEFAULT": 1,
    "height.NUCLEUS": 1,
    "phase.PROLIFERATIVE_G1": 0.5,
    "phase.PROLIFERATIVE_S": 0.5,
    "phase.PROLIFERATIVE_G2": 0.5,
    "phase.PROLIFERATIVE_M": 0.25,
    "phase.APOPTOTIC_EARLY": 0.25,
    "phase.APOPTOTIC_LATE": 0.5,
}

BIN_METRICS: list[str] = [
    "count",
    "volume",
    "height",
]

DISTRIBUTION_METRICS: list[str] = [
    "phase",
    "volume",
    "height",
]

INDIVIDUAL_METRICS: list[str] = [
    "volume",
    "height",
]

SPATIAL_METRICS: list[str] = [
    "population",
    "phase",
    "volume",
    "height",
]

TEMPORAL_METRICS: list[str] = [
    "count",
    "population",
    "phase",
    "volume",
    "height",
]


@dataclass
class ParametersConfigMetricsBins:
    """Parameter configuration for group basic metrics subflow - metrics bins."""

    metrics: list[str] = field(default_factory=lambda: BIN_METRICS)
    """List of bin metrics."""

    seed: int = 0
    """Simulation seed to use for grouping bin metrics."""

    ticks: list[int] = field(default_factory=lambda: [0])
    """Simulation ticks to use for grouping bin metrics."""

    scale: float = 1
    """Metric bin scaling."""

    ds: float = 1.0
    """Spatial scaling in units/um."""

    dt: float = 1.0
    """Temporal scaling in hours/tick."""


@dataclass
class ParametersConfigMetricsDistributions:
    """Parameter configuration for group basic metrics subflow - metrics distributions."""

    metrics: list[str] = field(default_factory=lambda: DISTRIBUTION_METRICS)
    """List of distribution metrics."""

    seeds: list[int] = field(default_factory=lambda: [0])
    """Simulation seeds to use for grouping metric distributions."""

    phases: list[str] = field(default_factory=lambda: CELL_PHASES)
    """List of cell cycle phases."""

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""

    ds: float = 1.0
    """Spatial scaling in units/um."""

    dt: float = 1.0
    """Temporal scaling in hours/tick."""

    bounds: dict[str, list] = field(default_factory=lambda: BOUNDS)
    """Bounds for metric distributions."""

    bandwidth: dict[str, float] = field(default_factory=lambda: BANDWIDTH)
    """Bandwidths for metric distributions."""


@dataclass
class ParametersConfigMetricsIndividuals:
    """Parameter configuration for group basic metrics subflow - metrics individuals."""

    metrics: list[str] = field(default_factory=lambda: INDIVIDUAL_METRICS)
    """List of individual metrics."""

    seed: int = 0
    """Simulation seed to use for grouping individual metrics."""

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""

    ds: float = 1.0
    """Spatial scaling in units/um."""

    dt: float = 1.0
    """Temporal scaling in hours/tick."""


@dataclass
class ParametersConfigMetricsSpatial:
    """Parameter configuration for group basic metrics subflow - metrics spatial."""

    metrics: list[str] = field(default_factory=lambda: SPATIAL_METRICS)
    """List of spatial metrics."""

    seeds: list[int] = field(default_factory=lambda: [0])
    """Simulation seeds to use for grouping spatial metrics."""

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""

    ds: float = 1.0
    """Spatial scaling in units/um."""

    dt: float = 1.0
    """Temporal scaling in hours/tick."""

    ticks: list[int] = field(default_factory=lambda: [0])
    """Simulation ticks to use for grouping spatial metrics."""


@dataclass
class ParametersConfigMetricsTemporal:
    """Parameter configuration for group basic metrics subflow - metrics temporal."""

    metrics: list[str] = field(default_factory=lambda: TEMPORAL_METRICS)
    """List of temporal metrics."""

    seeds: list[int] = field(default_factory=lambda: [0])
    """Simulation seeds to use for grouping temporal metrics."""

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""

    populations: list[int] = field(default_factory=lambda: [1])
    """List of cell populations."""

    phases: list[str] = field(default_factory=lambda: CELL_PHASES)
    """List of cell cycle phases."""

    ds: float = 1.0
    """Spatial scaling in units/um."""

    dt: float = 1.0
    """Temporal scaling in hours/tick."""


@dataclass
class ParametersConfigPopulationCounts:
    """Parameter configuration for group basic metrics subflow - population counts."""

    tick: int = 0
    """Simulation tick to use for grouping population counts."""

    seeds: list[int] = field(default_factory=lambda: [0])
    """Simulation seeds to use for grouping population counts."""


@dataclass
class ParametersConfig:
    """Parameter configuration for group basic metrics flow."""

    groups: list[str] = field(default_factory=lambda: GROUPS)
    """List of basic metric groups."""

    metrics_bins: ParametersConfigMetricsBins = ParametersConfigMetricsBins()
    """Parameters for group metrics bins subflow."""

    metrics_distributions: ParametersConfigMetricsDistributions = (
        ParametersConfigMetricsDistributions()
    )
    """Parameters for group metrics distributions subflow."""

    metrics_individuals: ParametersConfigMetricsIndividuals = ParametersConfigMetricsIndividuals()
    """Parameters for group metrics individuals subflow."""

    metrics_spatial: ParametersConfigMetricsSpatial = ParametersConfigMetricsSpatial()
    """Parameters for group metrics spatial subflow."""

    metrics_temporal: ParametersConfigMetricsTemporal = ParametersConfigMetricsTemporal()
    """Parameters for group metrics temporal subflow."""

    population_counts: ParametersConfigPopulationCounts = ParametersConfigPopulationCounts()
    """Parameters for group population counts subflow."""


@dataclass
class ContextConfig:
    """Context configuration for group basic metrics flow."""

    working_location: str
    """Location for input and output files (local path or S3 bucket)."""


@dataclass
class SeriesConfig:
    """Series configuration for group basic metrics flow."""

    name: str
    """Name of the simulation series."""

    conditions: list[dict]
    """List of series condition dictionaries (must include unique condition "key")."""


@flow(name="group-basic-metrics")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    """
    Main group basic metrics flow.

    Calls the following subflows, if the group is specified:

    - :py:func:`run_flow_group_metrics_bins`
    - :py:func:`run_flow_group_metrics_distributions`
    - :py:func:`run_flow_group_metrics_individuals`
    - :py:func:`run_flow_group_metrics_spatial`
    - :py:func:`run_flow_group_metrics_temporal`
    - :py:func:`run_flow_group_population_stats`
    """

    if "metrics_bins" in parameters.groups:
        run_flow_group_metrics_bins(context, series, parameters.metrics_bins)

    if "metrics_distributions" in parameters.groups:
        run_flow_group_metrics_distributions(context, series, parameters.metrics_distributions)

    if "metrics_individuals" in parameters.groups:
        run_flow_group_metrics_individuals(context, series, parameters.metrics_individuals)

    if "metrics_spatial" in parameters.groups:
        run_flow_group_metrics_spatial(context, series, parameters.metrics_spatial)

    if "metrics_temporal" in parameters.groups:
        run_flow_group_metrics_temporal(context, series, parameters.metrics_temporal)

    if "population_counts" in parameters.groups:
        run_flow_group_population_counts(context, series, parameters.population_counts)


@flow(name="group-basic-metrics_group-metrics-bins")
def run_flow_group_metrics_bins(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigMetricsBins
) -> None:
    """Group basic metrics subflow for binned metrics."""

    analysis_key = make_key(series.name, "analysis", "analysis.POSITIONS")
    group_key = make_key(series.name, "groups", "groups.BASIC")
    keys = [condition["key"] for condition in series.conditions]

    index_columns = ["x", "y"]

    for key in keys:
        series_key = f"{series.name}_{key}_{parameters.seed:04d}"

        results_key = make_key(series.name, "results", f"{series_key}.csv")
        results = load_dataframe(context.working_location, results_key)
        convert_model_units(results, parameters.ds, parameters.dt)

        positions_key = make_key(analysis_key, f"{series_key}.POSITIONS.csv")
        positions = load_dataframe(
            context.working_location, positions_key, converters={"id": ast.literal_eval}
        )

        for tick in parameters.ticks:
            tick_positions = positions[positions["TICK"] == tick]
            x = tick_positions["x"]
            y = tick_positions["y"]

            bins_df = pd.DataFrame()

            for metric in parameters.metrics:
                if metric == "count":
                    v = tick_positions["id"].map(len)
                else:
                    tick_results = results[results["TICK"] == tick].set_index("ID")
                    v = [
                        np.mean([tick_results.loc[i][metric] for i in ids])
                        for ids in tick_positions["id"]
                    ]

                bins = bin_to_hex(x, y, v, parameters.scale)
                bins_df_metric = pd.DataFrame(
                    [[x, y, np.mean(v)] for (x, y), v in bins.items()],
                    columns=index_columns + [metric.upper()],
                )

                if bins_df.empty:
                    bins_df = bins_df_metric
                else:
                    bins_df.set_index(index_columns, inplace=True)
                    bins_df_metric.set_index(index_columns, inplace=True)
                    bins_df = bins_df.join(bins_df_metric, on=index_columns)
                    bins_df = bins_df.reset_index()

            metric_key = f"{key}.{parameters.seed:04d}.{tick:06d}"
            save_dataframe(
                context.working_location,
                make_key(group_key, f"{series.name}.metrics_bins.{metric_key}.csv"),
                bins_df,
                index=False,
            )


@flow(name="group-basic-metrics_group-metrics-distributions")
def run_flow_group_metrics_distributions(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigMetricsDistributions
) -> None:
    """Group basic metrics subflow for distributions metrics."""

    logger = get_run_logger()
    group_key = make_key(series.name, "groups", "groups.BASIC")
    keys = [condition["key"] for condition in series.conditions]

    metrics: list[str] = []
    for metric in parameters.metrics:
        if metric in ["volume", "height"]:
            metrics = metrics + [f"{metric}.{region}" for region in parameters.regions]
        elif metric == "phase":
            metrics = metrics + [f"{metric}.{phase}" for phase in parameters.phases]
        else:
            continue

    distribution_bins: dict[str, dict] = {metric: {} for metric in metrics}
    distribution_means: dict[str, dict] = {metric: {} for metric in metrics}
    distribution_stdevs: dict[str, dict] = {metric: {} for metric in metrics}

    for key in keys:
        all_results = []

        for seed in parameters.seeds:
            series_key = f"{series.name}_{key}_{seed:04d}"
            results_key = make_key(series.name, "results", f"{series_key}.csv")

            if not check_key(context.working_location, results_key):
                logger.warning("[ %s ] seed [ %d ] not found", key, seed)
                continue

            results = load_dataframe(context.working_location, results_key)
            results["SEED"] = seed
            all_results.append(results)

        all_results_df = pd.concat(all_results)
        convert_model_units(all_results_df, parameters.ds, parameters.dt, parameters.regions)

        for metric in metrics:
            if "phase" in metric:
                phase = metric.split(".")[1]
                values = np.array(calculate_category_durations(all_results_df, "PHASE", phase))
            else:
                column = metric.replace(".DEFAULT", "")
                values = all_results_df[column].values

            bounds = (parameters.bounds[metric][0], parameters.bounds[metric][1])
            bandwidth = parameters.bandwidth[metric]

            distribution_means[metric][key] = np.mean(values)
            distribution_stdevs[metric][key] = np.std(values, ddof=1)
            distribution_bins[metric][key] = calculate_data_bins(values, bounds, bandwidth)

    for metric, distribution in distribution_bins.items():
        distribution["*"] = {
            "bandwidth": parameters.bandwidth[metric],
            "means": distribution_means[metric],
            "stdevs": distribution_stdevs[metric],
        }

        save_json(
            context.working_location,
            make_key(group_key, f"{series.name}.metrics_distributions.{metric.upper()}.json"),
            distribution,
        )


@flow(name="group-basic-metrics_group-metrics-individuals")
def run_flow_group_metrics_individuals(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigMetricsIndividuals
) -> None:
    """Group basic metrics subflow for individuals metrics."""

    group_key = make_key(series.name, "groups", "groups.BASIC")
    keys = [condition["key"] for condition in series.conditions]

    metrics: list[str] = [
        f"{metric}.{region}" for metric in parameters.metrics for region in parameters.regions
    ]

    for key in keys:
        series_key = f"{series.name}_{key}_{parameters.seed:04d}"
        results_key = make_key(series.name, "results", f"{series_key}.csv")
        results = load_dataframe(context.working_location, results_key)
        results.sort_values("TICK", inplace=True)
        convert_model_units(results, parameters.ds, parameters.dt, parameters.regions)

        for metric in metrics:
            times = results.groupby("ID")["time"].apply(np.hstack)
            values = results.groupby("ID")[metric.replace(".DEFAULT", "")].apply(np.hstack)
            phases = results.groupby("ID")["PHASE"].apply(np.hstack)

            entries = [
                [
                    {"time_and_value": np.array([x[:2] for x in group]), "phase": phase}
                    for phase, group in groupby(zip(time, value, phase), key=lambda x: x[2])
                ]
                for time, value, phase in zip(times, values, phases)
            ]

            individuals = [
                [
                    {
                        "time": item["time_and_value"][:, 0].tolist(),
                        "value": item["time_and_value"][:, 1].tolist(),
                        "phase": item["phase"],
                    }
                    for item in entry
                ]
                for entry in entries
            ]

            metric_key = f"{key}.{parameters.seed:04d}.{metric.upper()}"
            save_json(
                context.working_location,
                make_key(group_key, f"{series.name}.metrics_individuals.{metric_key}.json"),
                individuals,
            )


@flow(name="group-basic-metrics_group-metrics-spatial")
def run_flow_group_metrics_spatial(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigMetricsSpatial
) -> None:
    """Group basic metrics subflow for spatial metrics."""

    logger = get_run_logger()
    group_key = make_key(series.name, "groups", "groups.BASIC")
    keys = [condition["key"] for condition in series.conditions]

    metrics: list[str] = []
    for metric in parameters.metrics:
        if metric in ["volume", "height"]:
            metrics = metrics + [f"{metric}.{region}" for region in parameters.regions]
        else:
            metrics.append(metric)

    for key in keys:
        for seed in parameters.seeds:
            series_key = f"{series.name}_{key}_{seed:04d}"
            results_key = make_key(series.name, "results", f"{series_key}.csv")

            if not check_key(context.working_location, results_key):
                logger.warning("[ %s ] seed [ %d ] not found", key, seed)
                continue

            results = load_dataframe(context.working_location, results_key)
            convert_model_units(results, parameters.ds, parameters.dt, parameters.regions)

            for tick in parameters.ticks:
                tick_results = results[results["TICK"] == tick]

                for metric in metrics:
                    column = metric.replace(".DEFAULT", "") if "." in metric else metric.upper()
                    spatial = tick_results[["CENTER_X", "CENTER_Y", "CENTER_Z", column]].rename(
                        columns={"CENTER_X": "x", "CENTER_Y": "y", "CENTER_Z": "z", column: "v"}
                    )

                    metric_key = f"{key}.{seed:04d}.{tick:06d}.{metric.upper()}"
                    save_dataframe(
                        context.working_location,
                        make_key(group_key, f"{series.name}.metrics_spatial.{metric_key}.csv"),
                        spatial,
                        index=False,
                    )


@flow(name="group-basic-metrics_group-metrics-temporal")
def run_flow_group_metrics_temporal(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigMetricsTemporal
) -> None:
    """Group basic metrics subflow for temporal metrics."""

    logger = get_run_logger()
    group_key = make_key(series.name, "groups", "groups.BASIC")
    keys = [condition["key"] for condition in series.conditions]

    metrics: list[str] = []
    for metric in parameters.metrics:
        if metric in ["volume", "height"]:
            metrics = metrics + [f"{metric}.{region}" for region in parameters.regions]
        elif metric == "population":
            metrics = metrics + [f"{metric}.{population}" for population in parameters.populations]
        elif metric == "phase":
            metrics = metrics + [f"{metric}.{phase}" for phase in parameters.phases]
        else:
            metrics.append(metric)

    for key in keys:
        all_results = []

        for seed in parameters.seeds:
            series_key = f"{series.name}_{key}_{seed:04d}"
            results_key = make_key(series.name, "results", f"{series_key}.csv")

            if not check_key(context.working_location, results_key):
                logger.warning("[ %s ] seed [ %d ] not found", key, seed)
                continue

            results = load_dataframe(context.working_location, results_key)
            results["SEED"] = seed
            all_results.append(results)

        all_results_df = pd.concat(all_results)
        convert_model_units(all_results_df, parameters.ds, parameters.dt, parameters.regions)

        for metric in metrics:
            if metric == "count":
                values = all_results_df.groupby(["SEED", "time"]).size().groupby(["time"])
            elif "phase" in metric:
                phase_subset = all_results_df[all_results_df["PHASE"] == metric.split(".")[1]]
                phase_counts = phase_subset.groupby(["SEED", "time"]).size()
                total_counts = all_results_df.groupby(["SEED", "time"]).size()
                values = (phase_counts / total_counts).groupby("time")
            elif "population" in metric:
                pop_subset = all_results_df[
                    all_results_df["POPULATION"] == int(metric.split(".")[1])
                ]
                pop_counts = pop_subset.groupby(["SEED", "time"]).size()
                total_counts = all_results_df.groupby(["SEED", "time"]).size()
                values = (pop_counts / total_counts).groupby("time")
            else:
                column = metric.replace(".DEFAULT", "")
                values = all_results_df.groupby(["SEED", "time"])[column].mean().groupby(["time"])

            temporal = {
                "time": list(values.groups.keys()),
                "mean": [v if not np.isnan(v) else "nan" for v in values.mean()],
                "std": [v if not np.isnan(v) else "nan" for v in values.std(ddof=1)],
                "min": [v if not np.isnan(v) else "nan" for v in values.min()],
                "max": [v if not np.isnan(v) else "nan" for v in values.max()],
            }

            save_json(
                context.working_location,
                make_key(group_key, f"{series.name}.metrics_temporal.{key}.{metric.upper()}.json"),
                temporal,
            )


@flow(name="group-basic-metrics_group-population-counts")
def run_flow_group_population_counts(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigPopulationCounts
) -> None:
    """Group basic metrics subflow for population counts."""

    logger = get_run_logger()
    group_key = make_key(series.name, "groups", "groups.BASIC")
    keys = [condition["key"] for condition in series.conditions]

    counts: list[dict] = []

    for key in keys:
        for seed in parameters.seeds:
            series_key = f"{series.name}_{key}_{seed:04d}"
            results_key = make_key(series.name, "results", f"{series_key}.csv")

            if not check_key(context.working_location, results_key):
                logger.warning("[ %s ] seed [ %d ] not found", key, seed)
                continue

            results = load_dataframe(context.working_location, results_key, usecols=["TICK"])
            counts.append(
                {
                    "key": key,
                    "seed": seed,
                    "count": len(results[results["TICK"] == parameters.tick]),
                }
            )

    save_dataframe(
        context.working_location,
        make_key(group_key, f"{series.name}.population_counts.csv"),
        pd.DataFrame(counts),
        index=False,
    )
