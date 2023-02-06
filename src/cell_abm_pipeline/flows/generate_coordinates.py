"""
Workflow to generate cell ids and coordinates.

Working location structure:

.. code-block:: bash

    (name)
    ├── plots
    │    └── plots.COORDINATES
    │        └── (name).COORDINATES.png
    └── inits
        └── inits.COORDINATES
            └── (name).COORDINATES.csv

Generated coordinates are placed into the **inits/inits.COORDINATES** directory.
Corresponding plots are placed into the **plots/plots.COORDINATES** directory.
"""

from dataclasses import dataclass
from math import ceil, pi, sqrt
from typing import Tuple

from abm_initialization_collection.coordinate import filter_coordinate_bounds, make_grid_coordinates
from abm_initialization_collection.image import plot_contact_sheet
from io_collection.keys import make_key
from io_collection.save import save_dataframe, save_figure
from prefect import flow

AVERAGE_CELL_HEIGHT = 9.0

AVERAGE_CELL_VOLUME = 1300.0


@dataclass
class ParametersConfig:
    grid: str = "rect"

    ds: float = 1.0  # um/voxel

    bounding_box: Tuple[int, int] = (100, 100)  # um

    cell_height: float = AVERAGE_CELL_HEIGHT

    cell_volume: float = AVERAGE_CELL_VOLUME

    contact_sheet: bool = True


@dataclass
class ContextConfig:
    working_location: str


@dataclass
class SeriesConfig:
    name: str


@flow(name="generate-coordinates")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    # Calculate cell sizes (in elements)
    cell_radius = sqrt(parameters.cell_volume / parameters.cell_height / pi) / parameters.ds
    cell_bounds = (
        ceil(2 * cell_radius),
        ceil(2 * cell_radius),
        ceil(parameters.cell_height / parameters.ds),
    )

    # Make coordinates and filter for coordinates within cell radius.
    coordinates = make_grid_coordinates(parameters.grid, cell_bounds, 1, 1)
    coordinates = filter_coordinate_bounds(coordinates, cell_radius)

    # Recenter coordinates.
    coordinates["x"] = coordinates["x"] + parameters.bounding_box[0] / parameters.ds / 2
    coordinates["y"] = coordinates["y"] + parameters.bounding_box[1] / parameters.ds / 2

    # Placeholder id
    coordinates["id"] = 0

    # Save coordinates.
    coordinates_key = make_key(
        series.name, "inits", "inits.COORDINATES", f"{series.name}.COORDINATES.csv"
    )
    save_dataframe(context.working_location, coordinates_key, coordinates, index=False)

    if parameters.contact_sheet:
        contact_sheet = plot_contact_sheet(coordinates)
        plot_key = make_key(
            series.name, "plots", "plots.COORDINATES", f"{series.name}.COORDINATES.png"
        )
        save_figure(context.working_location, plot_key, contact_sheet)
