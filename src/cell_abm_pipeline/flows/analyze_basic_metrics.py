"""
Workflow for analyzing basic metrics.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arcade_collection.output import extract_tick_json, get_location_voxels
from io_collection.keys import check_key, make_key
from io_collection.load import load_dataframe, load_tar
from io_collection.save import save_dataframe
from prefect import flow


@dataclass
class ParametersConfig:
    """Parameter configuration for analyze basic metrics flow."""

    ticks: list[int]


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
    """Main analyze basic metrics flow."""

    run_flow_parse_positions(context, series, parameters)


@flow(name="analyze-basic-metrics_parse-positions")
def run_flow_parse_positions(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    data_key = make_key(series.name, "data", "data.LOCATIONS")
    analysis_key = make_key(series.name, "analysis", "analysis.POSITIONS")

    for condition in series.conditions:
        for seed in series.seeds:
            series_key = f"{series.name}_{condition['key']}_{seed:04d}"

            position_key = make_key(analysis_key, f"{series_key}.POSITIONS.csv")
            position_key_exists = check_key(context.working_location, position_key)

            existing_ticks = []
            if position_key_exists:
                existing_positions = load_dataframe(
                    context.working_location, position_key, usecols=["TICK"]
                )
                existing_ticks = list(existing_positions["TICK"].unique())

            tar_key = make_key(data_key, f"{series_key}.LOCATIONS.tar.xz")
            tar = load_tar(context.working_location, tar_key)

            for tick in parameters.ticks:
                if tick in existing_ticks:
                    continue

                tick_key = make_key(analysis_key, f"{series_key}_{tick:06d}.POSITIONS.csv")
                tick_key_exists = check_key(context.working_location, tick_key)

                if tick_key_exists:
                    continue

                locations = extract_tick_json(tar, series_key, tick, "LOCATIONS")
                positions = [
                    [x, y, location["id"]]
                    for location in locations
                    for x, y, _ in get_location_voxels.fn(location)
                ]
                positions_dataframe = pd.DataFrame(positions, columns=["x", "y", "id"])
                positions_unique = (
                    positions_dataframe.groupby(["x", "y"])["id"]
                    .apply(lambda x: list(np.unique(x)))
                    .reset_index()
                )
                positions_unique["TICK"] = tick

                save_dataframe(context.working_location, tick_key, positions_unique, index=False)
