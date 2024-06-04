"""
Workflow for calculating voxel positions.

Working location structure:

.. code-block:: bash

    (name)
    ├── data
    │   └── data.LOCATIONS
    │       └── (name)_(key)_(seed).LOCATIONS.tar.xz
    └── calculations
        └── calculations.POSITIONS
            └── (name)_(key)_(seed)_(tick).POSITIONS.csv

Data from **data.LOCATIONS** are used to calculate positions, which are saved to
**calculations.POSITIONS**.
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
    """Simulation key to calculate."""

    seed: int
    """Simulation random seed to calculate."""

    tick: int
    """Simulation tick to calculate."""


@dataclass
class ContextConfig:
    """Context configuration for calculate positions flow."""

    working_location: str
    """Location for input and output files (local path or S3 bucket)."""


@dataclass
class SeriesConfig:
    """Series configuration for calculate positions flow."""

    name: str
    """Name of the simulation series."""


@flow(name="calculate-positions")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    """Main calculate positions flow."""

    data_key = make_key(series.name, "data", "data.LOCATIONS")
    calc_key = make_key(series.name, "calculations", "calculations.POSITIONS")
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
        positions_dataframe.groupby(["x", "y"])["ids"]
        .apply(lambda x: list(np.unique(x)))
        .reset_index()
    )

    positions_unique["KEY"] = parameters.key
    positions_unique["SEED"] = parameters.seed
    positions_unique["TICK"] = parameters.tick

    positions_key = make_key(calc_key, f"{series_key}_{parameters.tick:06d}.POSITIONS.csv")
    save_dataframe(context.working_location, positions_key, positions_unique, index=False)
