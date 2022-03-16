from math import floor, sqrt

import numpy as np
import pandas as pd
from hexalattice.hexalattice import create_hex_grid
import matplotlib.pyplot as plt

from project_aics.initial_conditions.__config__ import SCALE_MICRONS_XY, SCALE_MICRONS_Z
from project_aics.utilities.load import load_image_from_fs
from project_aics.utilities.save import save_dataframe, save_image
from project_aics.utilities.keys import make_folder_key, make_file_key
from project_aics.utilities.plot import make_plot


class SampleImages:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "image": make_folder_key(context.name, "images", "", False),
            "sample": make_folder_key(context.name, "samples", "RAW", False),
            "contact": make_folder_key(context.name, "plots", "SAMPLE", False),
        }
        self.files = {
            "image": make_file_key(context.name, ["ome", "tiff"], "%s", ""),
            "sample": make_file_key(context.name, ["RAW", "csv"], "%s", ""),
            "contact": make_file_key(context.name, ["SAMPLE", "png"], "%s", ""),
        }

    def run(self, resolution=1, grid="rect", contact=True):
        for key in self.context.keys:
            df = self.sample_images(key, resolution, grid)

            if contact:
                self.plot_contact_sheet(key, df)

    def sample_images(self, key, resolution, grid):
        image_key = self.folders["image"] + self.files["image"] % (key)
        image = load_image_from_fs(self.context.working, image_key)

        sample_indices = self.get_sample_indices(image, resolution, grid)
        samples = self.get_image_samples(image, sample_indices)
        samples_df = pd.DataFrame(samples, columns=["id", "x", "y", "z"])

        sample_key = self.folders["sample"] + self.files["sample"] % (key)
        save_dataframe(self.context.working, sample_key, samples_df, index=False)

        return samples_df

    def plot_contact_sheet(self, key, data):
        """Save contact sheet image for all z slices in samples."""
        if len(data) == 0:
            return

        make_plot(sorted(data.z.unique()), data, self._plot_contact_sheet)

        plt.gca().invert_yaxis()
        plot_key = self.folders["contact"] + self.files["contact"] % key
        save_image(self.context.working, plot_key)

    @staticmethod
    def _plot_contact_sheet(ax, data, key):
        z_slice = data[data.z == key]

        max_id = int(data.id.max())
        min_id = int(data.id.min())

        ax.scatter(z_slice.x, z_slice.y, c=z_slice.id, vmin=min_id, vmax=max_id, s=1, cmap="jet")

    @staticmethod
    def get_sample_indices(image, resolution=1, grid="rect"):
        if grid == "rect":
            return SampleImages.get_rect_sample_indices(image, resolution)
        elif grid == "hex":
            return SampleImages.get_hex_sample_indices(image, resolution)
        else:
            raise ValueError(f"invalid grid type {grid}")

    @staticmethod
    def get_hex_sample_indices(image, resolution=1):
        """Get list of (x, y, z) sample indices for hex grid."""
        x_size, y_size, z_size = image.get_image_data("XYZ", S=0, T=0, C=1).shape

        z_increment = round(resolution / SCALE_MICRONS_Z)
        z_indices = np.arange(0, z_size, z_increment)
        z_offsets = [(i % 2) * (z_increment / 2) for i in range(len(z_indices))]

        xy_increment = round(resolution / SCALE_MICRONS_XY)
        xy_indicies, _ = create_hex_grid(
            nx=floor(x_size / xy_increment),
            ny=floor(y_size / xy_increment * sqrt(3)),
            min_diam=xy_increment,
            align_to_origin=False,
            do_plot=False,
        )

        sample_indices = [
            (round(x + offset), round(y + offset), z)
            for z, offset in zip(z_indices, z_offsets)
            for x, y in xy_indicies
            if round(x + offset) < x_size and round(y + offset) < y_size
        ]

        return sample_indices

    @staticmethod
    def get_rect_sample_indices(image, resolution=1):
        """Get list of (x, y, z) sample indices for rect grid."""
        x_size, y_size, z_size = image.get_image_data("XYZ", S=0, T=0, C=1).shape

        z_increment = round(resolution / SCALE_MICRONS_Z)
        z_indices = np.arange(0, z_size, z_increment)

        xy_increment = round(resolution / SCALE_MICRONS_XY)
        x_indices = np.arange(0, x_size, xy_increment)
        y_indices = np.arange(0, y_size, xy_increment)

        sample_indices = [(x, y, z) for z in z_indices for x in x_indices for y in y_indices]
        return sample_indices

    @staticmethod
    def get_image_samples(image, sample_indices, region=None):
        """Sample image at given indices."""
        array = image.get_image_data("XYZ", S=0, T=0, C=1)
        samples = [(array[x, y, z], x, y, z) for x, y, z in sample_indices if array[x, y, z] > 0]
        return samples
