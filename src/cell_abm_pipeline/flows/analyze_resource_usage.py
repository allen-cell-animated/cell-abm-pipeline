"""
Workflow for analyzing resource usage.
"""

import os
import re
from dataclasses import dataclass, field

import boto3
import pandas as pd
from io_collection.keys import check_key, get_keys, make_key
from io_collection.load import load_text
from io_collection.save import save_dataframe
from prefect import flow

STORAGE_GROUPS = ["CELLS", "LOCATIONS"]

STORAGE_PATTERN = r"[_]*([A-z0-9\s\_]*)_([0-9]{4})\."

CLOCK_PATTERN = r"simulation \[ ([A-z0-9\s\_]+) \| ([0-9]{4}) \] finished in ([0-9\.]+) minutes"


@dataclass
class ParametersConfig:
    """Parameter configuration for analyze resource usage flow."""

    groups: list[str] = field(default_factory=lambda: STORAGE_GROUPS)
    """Names of data storage groups."""

    search_locations: list[str] = field(default_factory=lambda: [])
    """List of locations (local paths or S3 buckets) to search for files."""

    exceptions: list[str] = field(default_factory=lambda: [])
    """List of exception strings used to filter log files."""


@dataclass
class ContextConfig:
    """Context configuration for analyze resource usage flow."""

    working_location: str
    """Location for input and output files (local path or S3 bucket)."""


@dataclass
class SeriesConfig:
    """Series configuration for analyze resource usage flow."""

    name: str
    """Name of the simulation series."""

    seeds: list[int]
    """List of series random seeds."""

    conditions: list[dict]
    """List of series condition dictionaries (must include unique condition "key")."""


@flow(name="analyze-resource-usage")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    """
    Main analyze resource usage flow.

    Calls the following subflows:

    - :py:func:`run_flow_analyze_storage`

    Iterate through simulation data files to extract storage size.

    - :py:func:`run_flow_analyze_clock`

    Iterate through simulation logs to extract wall clock time.
    """

    run_flow_analyze_storage(context, series, parameters)

    run_flow_analyze_clock(context, series, parameters)


@flow(name="analyze-resource-usage_analyze-storage")
def run_flow_analyze_storage(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    """Analyze resource usage subflow for analyzing storage size."""

    resources_key = make_key(series.name, "analysis", "analysis.RESOURCES")
    storage_key = make_key(resources_key, f"{series.name}_object_storage.RESOURCES.csv")

    if check_key(context.working_location, storage_key):
        return

    all_storage = []

    for group in parameters.groups:
        for location in parameters.search_locations:
            file_keys = get_keys(location, make_key(series.name, "data", f"data.{group}"))

            for file_key in file_keys:
                key, seed = re.findall(STORAGE_PATTERN, file_key.split(series.name)[-1])[0]

                if location.startswith("s3://"):
                    summary = boto3.resource("s3").ObjectSummary(location[5:], file_key)
                    storage = summary.size
                else:
                    storage = os.path.getsize(f"{location}{file_key}")

                all_storage.append({"KEY": key, "SEED": seed, "GROUP": group, "STORAGE": storage})

    storage_df = pd.DataFrame(all_storage)
    storage_df.sort_values(by=["KEY", "GROUP", "SEED"], ignore_index=True, inplace=True)

    save_dataframe(context.working_location, storage_key, storage_df, index=False)


@flow(name="analyze-resource-usage_analyze-clock")
def run_flow_analyze_clock(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    """Analyze resource usage subflow for analyzing wall clock time."""

    resources_key = make_key(series.name, "analysis", "analysis.RESOURCES")
    clock_key = make_key(resources_key, f"{series.name}_wall_clock.RESOURCES.csv")

    if check_key(context.working_location, clock_key):
        return

    all_clock = []

    for location in parameters.search_locations:
        file_keys = get_keys(location, make_key(series.name, "logs"))

        for file_key in file_keys:
            contents = load_text(location, file_key)

            if any(exception in contents for exception in parameters.exceptions):
                continue

            matches = re.findall(CLOCK_PATTERN, contents)

            for key, seed, clock in matches:
                all_clock.append({"KEY": key, "SEED": seed, "CLOCK": clock})

    clock_df = pd.DataFrame(all_clock)
    clock_df.sort_values(by=["KEY", "SEED"], ignore_index=True, inplace=True)

    save_dataframe(context.working_location, clock_key, clock_df, index=False)
