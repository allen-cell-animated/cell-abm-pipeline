from dataclasses import dataclass, field
from math import ceil, pi, sqrt

import pandas as pd
from abm_initialization_collection.coordinate import filter_coordinate_bounds, make_grid_coordinates
from abm_initialization_collection.image import plot_contact_sheet
from io_collection.keys import make_key
from io_collection.save import save_dataframe, save_figure
from prefect import flow

AVERAGE_CELL_HEIGHT = 9.0

AVERAGE_CELL_VOLUME = 1300.0

DEFAULT_CELL_ID = 1

SUBSTRATE_ID = -1


@dataclass
class ParametersConfig:
    grid: str = "rect"

    ds: list[float] = field(default_factory=lambda: [1.0])  # um/voxel

    bounding_box: tuple[int, int] = (100, 100)  # um

    cell_height: float = AVERAGE_CELL_HEIGHT

    cell_volume: float = AVERAGE_CELL_VOLUME

    substrate: bool = True

    contact_sheet: bool = True


@dataclass
class ContextConfig:
    working_location: str


@dataclass
class SeriesConfig:
    name: str


@flow(name="initialize-physicell-simulations")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    for ds in parameters.ds:
        cell_radius = sqrt(parameters.cell_volume / parameters.cell_height / pi)
        cell_height = parameters.cell_height

        # Set bounds for generating coordinates. If substrate is included, add
        # an additional z layer to make sure the cell has enough layers.
        if parameters.substrate:
            x_bound = ceil(parameters.bounding_box[0])
            y_bound = ceil(parameters.bounding_box[1])
            z_bound = ceil(cell_height + ds)
        else:
            x_bound = ceil(2 * cell_radius)
            y_bound = ceil(2 * cell_radius)
            z_bound = ceil(cell_height)

        # Generate the coordinates.
        coordinates_list = make_grid_coordinates(
            parameters.grid, (x_bound, y_bound, z_bound), ds, ds
        )

        # Filter coordinates for the specified cell radius and set default id.
        cell_coordinates = filter_coordinate_bounds(coordinates_list, cell_radius, center=False)
        cell_coordinates["id"] = DEFAULT_CELL_ID

        # If substrate is included, filter out the coordinates at z = 0 to use
        # as substrate coordinates with substrate id.
        if parameters.substrate:
            substrate_coordinates_list = [(x, y, z) for x, y, z in coordinates_list if z == 0]
            substrate_coordinates = pd.DataFrame(
                substrate_coordinates_list, columns=["x", "y", "z"]
            )
            substrate_coordinates["id"] = SUBSTRATE_ID
            cell_coordinates.drop(cell_coordinates[cell_coordinates["z"] == 0].index, inplace=True)
        else:
            substrate_coordinates = pd.DataFrame()

        # Save final list of coordinates.
        coordinates = pd.concat([substrate_coordinates, cell_coordinates])
        coordinates_key = make_key(
            series.name, "inits", "inits.PHYSICELL", f"{series.name}_{ds}.csv"
        )
        save_dataframe(
            context.working_location, coordinates_key, coordinates, index=False, header=False
        )

        # Plot contact sheet of coordinates.
        if parameters.contact_sheet:
            contact_sheet = plot_contact_sheet(coordinates)
            plot_key = make_key(
                series.name, "plots", "plots.COORDINATES", f"{series.name}_{ds}.COORDINATES.png"
            )
            save_figure(context.working_location, plot_key, contact_sheet)
