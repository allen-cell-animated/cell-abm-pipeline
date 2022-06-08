from math import floor, sqrt
from typing import List, Tuple

import numpy as np
import pandas as pd
from hexalattice.hexalattice import create_hex_grid
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from aicsimageio import AICSImage

from cell_abm_pipeline.initial_conditions.__config__ import SCALE_MICRONS_XY, SCALE_MICRONS_Z
from cell_abm_pipeline.initial_conditions.__main__ import Context
from cell_abm_pipeline.utilities.load import load_dataframe, load_image
from cell_abm_pipeline.utilities.save import save_dataframe, save_plot
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key, make_full_key
from cell_abm_pipeline.utilities.plot import make_plot


class SampleImages:
    """
    Task to sample ids and coordinates from images.

    Working location structure for a given context:

    .. code-block:: bash

        (name)/
        ├── images/
        │    ├── (name)_(image key 1).ome.tiff
        │    ├── (name)_(image key 2).ome.tiff
        │    ├── ...
        │    └── (name)_(image key n).ome.tiff
        ├── plots/
        │    └── plots.SAMPLE
        │        ├── (name)_(image key 1)_(channel).SAMPLE.png
        │        ├── (name)_(image key 2)_(channel).SAMPLE.png
        │        ├── ...
        │        └── (name)_(image key n)_(channel).SAMPLE.png
        └── samples/
            └── samples.RAW
                ├── (name)_(image key 1)_(channel).RAW.csv
                ├── (name)_(image key 2)_(channel).RAW.csv
                ├── ...
                └── (name)_(image key n)_(channel).RAW.csv

    The ``images`` directory contains the input images to be sampled.
    Resulting samples are placed into the ``samples/samples.RAW`` directory and
    corresponding plots are placed into the ```plots/plots.SAMPLE`` directory.

    Attributes
    ----------
    context
        ``Context`` object defining working location and name.
    folders
        Dictionary of input and output folder keys.
    files
        Dictionary of input and output file keys.
    """

    def __init__(self, context: Context):
        self.context = context
        self.folders = {
            "image": make_folder_key(context.name, "images", "", False),
            "sample": make_folder_key(context.name, "samples", "RAW", False),
            "contact": make_folder_key(context.name, "plots", "SAMPLE", False),
        }
        self.files = {
            "image": make_file_key(context.name, ["ome", "tiff"], "%s", ""),
            "sample": make_file_key(context.name, ["RAW", "csv"], "%s", "%02d"),
            "contact": make_file_key(context.name, ["SAMPLE", "png"], "%s", "%02d"),
        }

    def run(
        self,
        grid: str = "rect",
        resolution: float = 1.0,
        channels: List[int] = [0],
        contact: bool = True,
    ) -> None:
        """
        Runs sample images task for given context.

        Parameters
        ----------
        grid : {'rect', 'hex'}
            Type of sampling grid.
        resolution
            Distance between samples (um).
        channels
            Image channel indices.
        contact
            True if contact sheet of images is saved, False otherwise.
        """
        for key in self.context.keys:
            self.sample_images(key, grid, resolution, channels)

            if contact:
                for channel in channels:
                    self.plot_contact_sheet(key, channel)

    def sample_images(self, key: str, grid: str, resolution: float, channels: List[int]) -> None:
        """
        Sample image task.

        Loads image from working location.
        Calculates the sampling indices based on grid choice and image size.
        Samples of ids are taken from the image at each sampling index for each
        selected channel.
        Samples for each channel are saved separately.

        Parameters
        ----------
        key
            Key for image.
        grid : {'rect', 'hex'}
            Type of sampling grid.
        resolution
            Distance between samples (um).
        channels
            Image channel indices.
        """
        image_key = make_full_key(self.folders, self.files, "image", key)
        image = load_image(self.context.working, image_key)

        bounds = self.get_image_bounds(image)
        sample_indices = self.get_sample_indices(bounds, grid, resolution)

        for channel in channels:
            samples = self.get_image_samples(image, sample_indices, channel)
            samples_df = pd.DataFrame(samples, columns=["id", "x", "y", "z"])

            sample_key = make_full_key(self.folders, self.files, "sample", (key, channel))
            save_dataframe(self.context.working, sample_key, samples_df, index=False)

    def plot_contact_sheet(self, key: str, channel: int) -> None:
        """
        Plot contact sheet for selected image channel.

        Parameters
        ----------
        key
            Key for image.
        channel
            Image channel index.
        """
        data_key = make_full_key(self.folders, self.files, "sample", (key, channel))
        data = load_dataframe(self.context.working, data_key)

        if len(data) == 0:
            return

        make_plot(sorted(data.z.unique()), data, self.plot_contact_sheet_axes)

        plt.gca().invert_yaxis()
        plot_key = make_full_key(self.folders, self.files, "contact", (key, channel))
        save_plot(self.context.working, plot_key)

    @staticmethod
    def plot_contact_sheet_axes(ax: Axes, data: pd.DataFrame, key: int) -> None:
        """
        Plot samples for selected z index, colored by id.

        Parameters
        ----------
        ax
            ``Axes`` instance to plot on.
        data
            Image samples.
        key
            Index of z slice to plot.
        """
        z_slice = data[data["z"] == key]

        max_id = int(data.id.max())
        min_id = int(data.id.min())

        ax.scatter(z_slice.x, z_slice.y, c=z_slice.id, vmin=min_id, vmax=max_id, s=1, cmap="jet")
        ax.set_aspect("equal", adjustable="box")

    @staticmethod
    def get_image_bounds(image: AICSImage) -> Tuple[int, int, int]:
        """
        Extracts image bounds in the x, y, and z directions.

        Parameters
        ----------
        image
            Image object.

        Returns
        -------
        :
            Tuple of image bounds.
        """
        _, _, z_shape, y_shape, x_shape = image.shape
        bounds = (x_shape, y_shape, z_shape)
        return bounds

    @staticmethod
    def get_sample_indices(
        bounds: Tuple[int, int, int], grid: str = "rect", resolution: float = 1.0
    ) -> List:
        """
        Get sample indices with given bounds for selected grid type.

        Parameters
        ----------
        bounds
            Sampling bounds in the x, y, and z directions.
        grid : {'rect', 'hex'}
            Type of sampling grid.
        resolution
            Distance between samples (um).

        Returns
        -------
        :
            List of sample indices.
        """
        if grid == "rect":
            return SampleImages.get_rect_sample_indices(bounds, resolution)

        if grid == "hex":
            return SampleImages.get_hex_sample_indices(bounds, resolution)

        raise ValueError(f"invalid grid type {grid}")

    @staticmethod
    def get_hex_sample_indices(bounds: Tuple[int, int, int], resolution: float = 1.0) -> List:
        """
        Get list of (x, y, z) sample indices for hex grid.

        Parameters
        ----------
        bounds
            Sampling bounds in the x, y, and z directions.
        resolution
            Distance between samples (um).

        Returns
        -------
        :
            List of sample indices.
        """
        x_bound, y_bound, z_bound = bounds

        z_increment = round(resolution / SCALE_MICRONS_Z)
        z_indices = np.arange(0, z_bound, z_increment)
        z_offsets = [(i % 2) * (z_increment / 2) for i in range(len(z_indices))]

        xy_increment = round(resolution / SCALE_MICRONS_XY)
        xy_indices, _ = create_hex_grid(
            nx=floor(x_bound / xy_increment),
            ny=floor(y_bound / xy_increment * sqrt(3)),
            min_diam=xy_increment,
            align_to_origin=False,
            do_plot=False,
        )

        sample_indices = [
            (round(x + offset), round(y + offset), z)
            for z, offset in zip(z_indices, z_offsets)
            for x, y in xy_indices
            if round(x + offset) < x_bound and round(y + offset) < y_bound
        ]

        return sample_indices

    @staticmethod
    def get_rect_sample_indices(bounds: Tuple[int, int, int], resolution: float = 1.0) -> List:
        """
        Get list of (x, y, z) sample indices for rect grid.

        Parameters
        ----------
        bounds
            Sampling bounds in the x, y, and z directions.
        resolution
            Distance between samples (um).

        Returns
        -------
        :
            List of sample indices.
        """
        x_bound, y_bound, z_bound = bounds

        z_increment = round(resolution / SCALE_MICRONS_Z)
        z_indices = np.arange(0, z_bound, z_increment)

        xy_increment = round(resolution / SCALE_MICRONS_XY)
        x_indices = np.arange(0, x_bound, xy_increment)
        y_indices = np.arange(0, y_bound, xy_increment)

        sample_indices = [(x, y, z) for z in z_indices for x in x_indices for y in y_indices]
        return sample_indices

    @staticmethod
    def get_image_samples(image: AICSImage, sample_indices: List, channel: int = 0) -> List:
        """
        Sample image at given indices into list of (id, x, y, z) samples.

        Parameters
        ----------
        image
            Image object to sample.
        sample_indices
            List of sampling indices.
        channel
            Image channel to sample.

        Returns
        -------
        :
            List of image samples.
        """
        array = image.get_image_data("XYZ", T=0, C=channel)
        samples = [(array[x, y, z], x, y, z) for x, y, z in sample_indices if array[x, y, z] > 0]
        return samples
