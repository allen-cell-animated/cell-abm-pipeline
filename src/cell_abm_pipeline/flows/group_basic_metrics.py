"""
Workflow for grouping basic metrics.

Working location structure:

.. code-block:: bash

    (name)
    ├── groups
    │   └── groups.BASIC
    │       ├── (name).feature_distributions.(metric).json
    │       ├── (name).feature_distributions.(metric).json
    │       ├── ...
    │       ├── (name).feature_distributions.(metric).json
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
    │       └── (name).metrics_temporal.(key).(metric).json
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

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from arcade_collection.output import convert_model_units
from io_collection.keys import make_key
from io_collection.load import load_dataframe
from io_collection.save import save_dataframe, save_json
from prefect import flow

from cell_abm_pipeline.tasks import calculate_category_durations, calculate_data_bins

GROUPS: list[str] = [
    "metrics_distributions",
    "metrics_individuals",
    "metrics_spatial",
    "metrics_temporal",
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
    "phase.PROLIFERATIVE_G1": [0, 7],
    "phase.PROLIFERATIVE_S": [0, 21],
    "phase.PROLIFERATIVE_G2": [0, 20],
    "phase.PROLIFERATIVE_M": [0, 20],
    "phase.APOPTOTIC_EARLY": [0, 10],
    "phase.APOPTOTIC_LATE": [0, 10],
}

BANDWIDTH: dict[str, float] = {
    "volume.DEFAULT": 100,
    "volume.NUCLEUS": 50,
    "height.DEFAULT": 1,
    "height.NUCLEUS": 1,
    "phase.PROLIFERATIVE_G1": 1,
    "phase.PROLIFERATIVE_S": 1,
    "phase.PROLIFERATIVE_G2": 1,
    "phase.PROLIFERATIVE_M": 1,
    "phase.APOPTOTIC_EARLY": 1,
    "phase.APOPTOTIC_LATE": 1,
}

DISTRIBUTION_METRICS: list[str] = [
    "phase",
    "volume",
    "height",
]

INDIVIDUAL_METRICS: list[str] = [
    "phase",
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
class ParametersConfigMetricsDistributions:
    """Parameter configuration for group basic metrics subflow - metrics distributions."""

    metrics: list[str] = field(default_factory=lambda: DISTRIBUTION_METRICS)
    """List of spatial metrics."""

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
    """List of spatial metrics."""

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
class ParametersConfig:
    """Parameter configuration for group basic metrics flow."""

    groups: list[str] = field(default_factory=lambda: GROUPS)
    """List of basic metric groups."""

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

    seeds: list[int]
    """List of series random seeds."""

    conditions: list[dict]
    """List of series condition dictionaries (must include unique condition "key")."""


@flow(name="group-basic-metrics")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    """
    Main group basic metrics flow.

    Calls the following subflows, if the group is specified:

    - :py:func:`run_flow_group_metrics_distributions`
    - :py:func:`run_flow_group_metrics_individuals`
    - :py:func:`run_flow_group_metrics_spatial`
    - :py:func:`run_flow_group_metrics_temporal`
    """

    if "metrics_distributions" in parameters.groups:
        run_flow_group_metrics_distributions(context, series, parameters.metrics_distributions)

    if "metrics_individuals" in parameters.groups:
        run_flow_group_metrics_individuals(context, series, parameters.metrics_individuals)

    if "metrics_spatial" in parameters.groups:
        run_flow_group_metrics_spatial(context, series, parameters.metrics_spatial)

    if "metrics_temporal" in parameters.groups:
        run_flow_group_metrics_temporal(context, series, parameters.metrics_temporal)


@flow(name="group-basic-metrics_group-metrics-distributions")
def run_flow_group_metrics_distributions(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigMetricsDistributions
) -> None:
    """Group basic metrics subflow for distributions metrics."""

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

        for seed in series.seeds:
            series_key = f"{series.name}_{key}_{seed:04d}"
            results_key = make_key(series.name, "results", f"{series_key}.csv")
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
            make_key(group_key, f"{series.name}.feature_distributions.{metric.upper()}.json"),
            distribution,
        )


@flow(name="group-basic-metrics_group-metrics-individuals")
def run_flow_group_metrics_individuals(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigMetricsIndividuals
) -> None:
    """Group basic metrics subflow for individuals metrics."""

    group_key = make_key(series.name, "groups", "groups.BASIC")
    keys = [condition["key"] for condition in series.conditions]

    metrics: list[str] = []
    for metric in parameters.metrics:
        if metric in ["volume", "height"]:
            metrics = metrics + [f"{metric}.{region}" for region in parameters.regions]
        else:
            metrics.append(metric)

    for key in keys:
        for seed in series.seeds:
            series_key = f"{series.name}_{key}_{seed:04d}"
            results_key = make_key(series.name, "results", f"{series_key}.csv")
            results = load_dataframe(context.working_location, results_key)
            convert_model_units(results, parameters.ds, parameters.dt, parameters.regions)

            for metric in metrics:
                column = metric.replace(".DEFAULT", "") if "." in metric else metric.upper()
                times = results.groupby("ID")["time"].apply(np.hstack)
                values = results.groupby("ID")[column].apply(np.hstack)
                individuals = [
                    {"time": time.tolist(), "value": value.tolist()}
                    for time, value in zip(times, values)
                ]

                metric_key = f"{key}.{seed:04d}.{metric.upper()}"
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

    group_key = make_key(series.name, "groups", "groups.BASIC")
    keys = [condition["key"] for condition in series.conditions]

    metrics: list[str] = []
    for metric in parameters.metrics:
        if metric in ["volume", "height"]:
            metrics = metrics + [f"{metric}.{region}" for region in parameters.regions]
        else:
            metrics.append(metric)

    for key in keys:
        for seed in series.seeds:
            series_key = f"{series.name}_{key}_{seed:04d}"
            results_key = make_key(series.name, "results", f"{series_key}.csv")
            results = load_dataframe(context.working_location, results_key)
            convert_model_units(results, parameters.ds, parameters.dt, parameters.regions)

            for tick in parameters.ticks:
                tick_results = results[results["TICK"] == tick]

                for metric in metrics:
                    column = metric.replace(".DEFAULT", "") if "." in metric else metric.upper()
                    spatial = tick_results[["CENTER_X", "CENTER_Y", "CENTER_Z", column]]

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

        for seed in series.seeds:
            series_key = f"{series.name}_{key}_{seed:04d}"
            results_key = make_key(series.name, "results", f"{series_key}.csv")
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
                "mean": values.mean().values.tolist(),
                "std": values.std(ddof=1).values.tolist(),
                "min": values.min().values.tolist(),
                "max": values.max().values.tolist(),
            }

            save_json(
                context.working_location,
                make_key(group_key, f"{series.name}.metrics_temporal.{key}.{metric.upper()}.json"),
                temporal,
            )
