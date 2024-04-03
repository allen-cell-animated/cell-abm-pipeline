"""
Workflow for analyzing basic metrics.

Working location structure:

.. code-block:: bash

    (name)
    ├── analysis
    │   └── analysis.BASIC_METRICS
    │       └── (name)_(key).BASIC_METRICS.csv
    └── results
        └── (name)_(key)_(seed).csv

Data from **results** are processed into the **analysis.BASIC_METRICS** directory.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from itertools import groupby
from typing import Optional

import pandas as pd
from arcade_collection.output import convert_model_units
from io_collection.keys import check_key, make_key
from io_collection.load import load_dataframe
from io_collection.save import save_dataframe
from prefect import flow, get_run_logger
from prefect.tasks import task_input_hash

OPTIONS = {
    "cache_result_in_memory": False,
    "cache_key_fn": task_input_hash,
    "cache_expiration": timedelta(hours=12),
}


@dataclass
class ParametersConfig:
    """Parameter configuration for analyze basic metrics flow."""

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""

    ds: Optional[float] = None
    """Spatial scaling in units/um."""

    dt: Optional[float] = None
    """Temporal scaling in hours/tick."""

    full: bool = False
    """True if all conditions should be combined into a single dataset, False otherwise."""


@dataclass
class ContextConfig:
    """Context configuration for analyze basic metrics flow."""

    working_location: str
    """Location for input and output files (local path or S3 bucket)."""


@dataclass
class SeriesConfig:
    """Series configuration for analyze basic metrics flow."""

    name: str
    """Name of the simulation series."""

    seeds: list[int]
    """List of series random seeds."""

    conditions: list[dict]
    """List of series condition dictionaries (must include unique condition "key")."""


@flow(name="analyze-basic-metrics")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    """
    Main analyze basic metrics flow.

    Calls the following subflows, in order:

    1. :py:func:`run_flow_process_results`
    """

    run_flow_process_results(context, series, parameters)


@flow(name="analyze-basic-metrics_process-results")
def run_flow_process_results(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    """
    Analyze basic metrics subflow for processing results.

    Processes parsed simulation results and compiles into a single dataframe. If
    the combined dataframe already exists for a given key, that key is skipped.
    """

    logger = get_run_logger()

    results_path_key = make_key(series.name, "results")
    metrics_path_key = make_key(series.name, "analysis", "analysis.BASIC_METRICS")

    if parameters.full:
        superkeys = {"": [condition["key"] for condition in series.conditions]}
    else:
        keys = [condition["key"].split("_") for condition in series.conditions]
        superkeys = {
            superkey: ["_".join(k) for k in key_group]
            for index in range(len(keys[0]))
            for superkey, key_group in groupby(
                sorted(keys, key=lambda k: k[index]), lambda k: k[index]
            )
        }

    for superkey, key_group in superkeys.items():
        logger.info("Processing results for superkey [ %s ]", superkey)
        metrics_key = make_key(metrics_path_key, f"{series.name}_{superkey}.BASIC_METRICS.csv")

        if check_key(context.working_location, metrics_key):
            continue

        all_results = []

        for key in key_group:
            for seed in series.seeds:
                results_key = make_key(results_path_key, f"{series.name}_{key}_{seed:04d}.csv")
                results = load_dataframe.with_options(**OPTIONS)(
                    context.working_location, results_key
                )
                results["KEY"] = key
                results["SEED"] = seed
                all_results.append(results)

        # Combine into single dataframe.
        results_df = pd.concat(all_results)

        # Convert units.
        convert_model_units(results_df, parameters.ds, parameters.dt, parameters.regions)

        # Save final dataframe.
        save_dataframe(context.working_location, metrics_key, results_df, index=False)
