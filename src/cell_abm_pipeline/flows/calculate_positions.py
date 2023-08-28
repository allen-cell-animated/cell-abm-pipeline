"""
Workflow for calculating voxel positions.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arcade_collection.output import extract_tick_json, get_location_voxels
from io_collection.keys import make_key
from io_collection.load import load_tar
from io_collection.save import save_dataframe
from prefect import flow


@dataclass
class ParametersConfig:
    """Parameter configuration for calculate positions flow."""

    key: str

    seed: int

    tick: int


@dataclass
class ContextConfig:
    """Context configuration for calculate positions flow."""

    working_location: str


@dataclass
class SeriesConfig:
    """Series configuration for calculate positions flow."""

    name: str


@flow(name="calculate-positions")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    """Main calculate positions flow."""

    data_key = make_key(series.name, "data", "data.LOCATIONS")
    analysis_key = make_key(series.name, "analysis", "analysis.POSITIONS")
    series_key = f"{series.name}_{parameters.key}_{parameters.seed:04d}"

    locations_key = make_key(data_key, f"{series_key}.LOCATIONS.tar.xz")
    locations_tar = load_tar(context.working_location, locations_key)
    locations_json = extract_tick_json(locations_tar, series_key, parameters.tick, "LOCATIONS")

    positions = [
        [x, y, location["id"]]
        for location in locations_json
        for x, y, _ in get_location_voxels.fn(location)
    ]
    positions_dataframe = pd.DataFrame(positions, columns=["x", "y", "ids"])
    positions_unique = (
        positions_dataframe.groupby(["x", "y"])["id"]
        .apply(lambda x: list(np.unique(x)))
        .reset_index()
    )

    positions_unique["KEY"] = parameters.key
    positions_unique["SEED"] = parameters.seed
    positions_unique["TICK"] = parameters.tick

    positions_key = make_key(analysis_key, f"{series_key}_{parameters.tick:06d}.POSITIONS.csv")
    save_dataframe(context.working_location, positions_key, positions_unique, index=False)
