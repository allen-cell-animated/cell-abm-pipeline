from math import floor, pi, sqrt, ceil
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from hexalattice.hexalattice import create_hex_grid

from cell_abm_pipeline.initial_conditions.__config__ import (
    CRITICAL_VOLUME_AVGS,
    CRITICAL_HEIGHT_AVGS,
)
from cell_abm_pipeline.initial_conditions.__main__ import Context
from cell_abm_pipeline.utilities.load import load_dataframe
from cell_abm_pipeline.utilities.save import save_dataframe, save_plot
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key, make_full_key
from cell_abm_pipeline.utilities.plot import make_plot


class GenerateCoordinates:
    """
    Task to generate cell ids and coordinates.

    Working location structure for a given context:

    .. code-block:: bash

        (name)
        ├── plots
        │    └── plots.COORDINATES
        │        ├── (name)_(image key 1).COORDINATES.png
        │        ├── (name)_(image key 2).COORDINATES.png
        │        ├── ...
        │        └── (name)_(image key n).COORDINATES.png
        └── inits
            └── inits.COORDINATES
                ├── (name)_(image key 1).COORDINATES.csv
                ├── (name)_(image key 2).COORDINATES.csv
                ├── ...
                └── (name)_(image key n).COORDINATES.csv

    Generated coordinates are placed into the **inits/inits.GENERATED** directory.
    Corresponding plots are placed into the **plots/plots.GENERATED** directory.

    Attributes
    ----------
    context
        **Context** object defining working location and name.
    folders
        Dictionary of input and output folder keys.
    files
        Dictionary of input and output file keys.
    """

    def __init__(self, context: Context):
        self.context = context
        self.folders = {
            "coordinates": make_folder_key(context.name, "inits", "COORDINATES", False),
            "contact": make_folder_key(context.name, "plots", "COORDINATES", False),
        }
        self.files = {
            "coordinates": make_file_key(context.name, ["COORDINATES", "csv"], "%s", ""),
            "contact": make_file_key(context.name, ["COORDINATES", "png"], "%s", ""),
        }

    def run(
        self,
        grid: str = "rect",
        ds: float = 1.0,
        box: Tuple[int, int, int] = (100, 100, 10),
        contact: bool = True,
    ) -> None:
        """
        Runs generate coordinates task for given context.

        Parameters
        ----------
        grid : {'rect', 'hex'}
            Type of sampling grid.
        ds
            Distance between elements in um.
        box
            Bounding box size in um.
        contact
            True if contact sheet of coordinates is saved, False otherwise.
        """
        for key in self.context.keys:
            self.generate_coordinates(key, grid, ds, box)

            if contact:
                self.plot_contact_sheet(key)

    def generate_coordinates(
        self, key: str, grid: str, ds: float, box: Tuple[int, int, int]
    ) -> None:
        # Calculate cell sizes (in um).
        cell_height = CRITICAL_HEIGHT_AVGS["DEFAULT"] / ds
        cell_volume = CRITICAL_VOLUME_AVGS["DEFAULT"]
        cell_radius = sqrt(cell_volume / cell_height / pi) / ds
        cell_bounds = (ceil(2 * cell_radius), ceil(2 * cell_radius), ceil(cell_height))

        # Calculate cell center.
        length, width, _ = [box_dim / ds for box_dim in box]
        center = (length / 2, width / 2)

        # Generate coordinates for cell around centroid.
        coordinates = self.get_box_coordinates(cell_bounds, grid)
        coordinates = self.transform_cell_coordinates(coordinates, cell_radius, center)

        # Convert to dataframe and save.
        coordinates_df = pd.DataFrame(coordinates, columns=["x", "y", "z"])
        coordinates_df["id"] = 1
        coordinates_key = make_full_key(self.folders, self.files, "coordinates", key)
        save_dataframe(self.context.working, coordinates_key, coordinates_df, index=False)

    def plot_contact_sheet(self, key: str) -> None:
        """
        Plot contact sheet for generated coordinates.

        Parameters
        ----------
        key
            Key for coordinates.
        """
        coordinates_key = make_full_key(self.folders, self.files, "coordinates", key)
        data = load_dataframe(self.context.working, coordinates_key)

        make_plot(sorted(data.z.unique()), data, self.plot_contact_sheet_axes)

        plt.gca().invert_yaxis()
        plot_key = make_full_key(self.folders, self.files, "contact", key)
        save_plot(self.context.working, plot_key)

    @staticmethod
    def plot_contact_sheet_axes(ax: Axes, data: pd.DataFrame, key: int) -> None:
        """
        Plot coordinates for selected z index, colored by id.

        Parameters
        ----------
        ax
            **Axes** instance to plot on.
        data
            Coordinates data.
        key
            Index of z slice to plot.
        """
        z_slice = data[data.z == key]

        max_id = int(data.id.max())
        min_id = int(data.id.min())

        ax.scatter(z_slice.x, z_slice.y, c=z_slice.id, vmin=min_id, vmax=max_id, s=1, cmap="jet")
        ax.set_aspect("equal", adjustable="box")

    @staticmethod
    def get_box_coordinates(
        bounds: Tuple[int, int, int],
        grid: str = "rect",
    ) -> List:
        """
        Get all possible coordinates within given bounding box.

        Parameters
        ----------
        bounds
            Bounds in the x, y, and z directions.
        grid : {'rect', 'hex'}
            Type of sampling grid.

        Returns
        -------
        :
            List of grid coordinates.
        """
        if grid == "rect":
            return GenerateCoordinates.make_rect_coordinates(bounds, 1, 1)

        if grid == "hex":
            return GenerateCoordinates.make_hex_coordinates(bounds, 1, 1)

        raise ValueError(f"invalid grid type {grid}")

    @staticmethod
    def make_rect_coordinates(
        bounds: Tuple[int, int, int],
        xy_increment: float,
        z_increment: float,
    ) -> List:
        """
        Get list of bounded (x, y, z) coordinates for rect grid.

        Parameters
        ----------
        bounds
            Bounds in the x, y, and z directions.
        xy_increment
            Increment size in x/y.
        z_increment
            Increment size in z.

        Returns
        -------
        :
            List of grid coordinates.
        """
        x_bound, y_bound, z_bound = bounds

        z_indices = np.arange(0, z_bound, z_increment)
        x_indices = np.arange(0, x_bound, xy_increment)
        y_indices = np.arange(0, y_bound, xy_increment)

        coordinates = [(x, y, z) for z in z_indices for x in x_indices for y in y_indices]

        return coordinates

    @staticmethod
    def make_hex_coordinates(
        bounds: Tuple[int, int, int],
        xy_increment: float,
        z_increment: float,
    ) -> List:
        """
        Get list of bounded (x, y, z) coordinates for hex grid.

        Coordinates are offset in sets of three z slices to form a face-centered
        cubic (FCC) packing.

        Parameters
        ----------
        bounds
            Bounds in the x, y, and z directions.
        xy_increment
            Increment size in x/y.
        z_increment
            Increment size in z.

        Returns
        -------
        :
            List of grid coordinates.
        """
        x_bound, y_bound, z_bound = bounds

        z_indices = np.arange(0, z_bound, z_increment)
        z_offsets = [(i % 3) for i in range(len(z_indices))]

        xy_indices, _ = create_hex_grid(
            nx=floor(x_bound / xy_increment),
            ny=floor(y_bound / xy_increment * sqrt(3)),
            min_diam=xy_increment,
            align_to_origin=False,
            do_plot=False,
        )

        x_offsets = [(xy_increment / 2) if z_offset == 1 else 0 for z_offset in z_offsets]
        y_offsets = [(xy_increment / 2) * sqrt(3) / 3 * z_offset for z_offset in z_offsets]

        coordinates = [
            (x + x_offset, y + y_offset, z)
            for z, x_offset, y_offset in zip(z_indices, x_offsets, y_offsets)
            for x, y in xy_indices
            if round(x + x_offset) < x_bound and round(y + y_offset) < y_bound
        ]

        return coordinates

    @staticmethod
    def transform_cell_coordinates(
        coordinates: List, radius: float, offsets: Tuple[float, float]
    ) -> List:
        """
        Filters list for coordinates with given radius and applies offset.

        Parameters
        ----------
        coordinates
            List of (x, y, z) coordinates.
        radius
            Maximum valid radius of coordinate.
        offsets
            Coordinate offsets in the x and y directions.

        Returns
        -------
        :
            Filtered list of coordinates.
        """
        dx, dy = offsets
        filtered_coordinates = []

        for x, y, z in coordinates:
            coordinate_radius = (x - radius) ** 2 + (y - radius) ** 2
            if coordinate_radius <= radius**2:
                filtered_coordinates.append((x - radius + dx, y - radius + dy, z))

        return filtered_coordinates
