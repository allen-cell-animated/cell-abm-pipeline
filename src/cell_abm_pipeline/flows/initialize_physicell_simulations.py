"""
Workflow for initializing PhysiCell simulations.

Working location structure:

.. code-block:: bash

    (name)
    ├── inits
    │   └── inits.PHYSICELL
    │       └── (name)_(key)_(resolution).csv
    └── plots
        └── plots.COORDINATES
            └── (name)_(key)_(resolution).COORDINATES.png

Initialization consist of a single cell of the specified height and volume,
sampled at the given spatial resolution. Coordinates are saved to
**inits.PHYSICELL**.
"""

from dataclasses import dataclass, field
from math import ceil, pi, sqrt

import pandas as pd
from abm_initialization_collection.coordinate import filter_coordinate_bounds, make_grid_coordinates
from abm_initialization_collection.image import plot_contact_sheet
from io_collection.keys import make_key
from io_collection.save import save_dataframe, save_figure
from prefect import flow

# Default average cell height in um.
AVERAGE_CELL_HEIGHT = 9.0

# Default average cell volume in um^3.
AVERAGE_CELL_VOLUME = 1300.0

# Default cell id.
DEFAULT_CELL_ID = 1

# Substrate id.
SUBSTRATE_ID = -1


@dataclass
class ParametersConfig:
    """Parameter configuration for initialize PhysiCell simulations flow."""

    grid: str = "rect"
    """Type of sampling grid (rect = rectangular, hex = hexagonal)."""

    ds: list[float] = field(default_factory=lambda: [1.0])
    """Spatial scaling in um/voxel."""

    bounding_box: tuple[int, int] = (100, 100)
    """Size of bounding box in um."""

    cell_height: float = AVERAGE_CELL_HEIGHT
    """Average cell height in um."""

    cell_volume: float = AVERAGE_CELL_VOLUME
    """Average cell volume in um^3."""

    substrate: bool = True
    """True to include substrate in initialization, False otherwise."""

    contact_sheet: bool = True
    """True to save contact sheet of initialization, False otherwise."""


@dataclass
class ContextConfig:
    """Context configuration for initialize PhysiCell simulations flow."""

    working_location: str
    """Location for input and output files (local path or S3 bucket)."""


@dataclass
class SeriesConfig:
    """Series configuration for initialize PhysiCell simulations flow."""

    name: str
    """Name of the simulation series."""


@flow(name="initialize-physicell-simulations")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    """Main initialize PhysiCell simulations flow."""

    for ds in parameters.ds:
        # Calculate cell radius and height.
        cell_radius = sqrt(parameters.cell_volume / parameters.cell_height / pi)
        cell_height = parameters.cell_height

        # Adjust size of bounding box.
        x_bound = ceil(parameters.bounding_box[0] / ds) * ds
        y_bound = ceil(parameters.bounding_box[1] / ds) * ds
        z_bound = ceil(cell_height)

        # Generate full coordinates list.
        grid_bounds = (ceil(x_bound + ds), ceil(y_bound + ds), ceil(z_bound + ds))
        coords_list = make_grid_coordinates(parameters.grid, grid_bounds, ds, ds)

        # Filter coordinates list for cell coordinates.
        cell_coords_list = [
            (x, y, z)
            for x, y, z in coords_list
            if x_bound / 2 - cell_radius <= x <= x_bound / 2 + cell_radius
            and y_bound / 2 - cell_radius <= y <= y_bound / 2 + cell_radius
            and z > 0
        ]
        cell_coords = filter_coordinate_bounds(cell_coords_list, cell_radius, center=False)
        cell_coords["id"] = DEFAULT_CELL_ID

        # If substrate is included, add the z = 0 coordinates. If not, adjust
        # all the z positions of the cell coordinates.
        if parameters.substrate:
            substrate_coords_list = [(x, y, z) for x, y, z in coords_list if z == 0]
            substrate_coords = pd.DataFrame(substrate_coords_list, columns=["x", "y", "z"])
            substrate_coords["id"] = SUBSTRATE_ID
        else:
            cell_coords["z"] = cell_coords["z"] - ds
            substrate_coords = pd.DataFrame()

        # Save final list of coordinates.
        coords = pd.concat([substrate_coords, cell_coords])
        init_key = make_key(series.name, "inits", "inits.PHYSICELL", f"{series.name}_{ds}.csv")
        save_dataframe(context.working_location, init_key, coords, index=False, header=False)

        # Plot contact sheet of coordinates.
        if parameters.contact_sheet:
            contact_sheet = plot_contact_sheet(coords)
            plot_key = make_key(
                series.name, "plots", "plots.COORDINATES", f"{series.name}_{ds}.COORDINATES.png"
            )
            save_figure(context.working_location, plot_key, contact_sheet)
