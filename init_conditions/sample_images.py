import os
import math
import imageio
import numpy as np
import pandas as pd
from glob import glob
from hexalattice.hexalattice import create_hex_grid
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from constants import SCALE_MICRONS, SCALE_MICRONS_Z, OUTPUT_COLUMNS


class SampleImages:
    def __init__(self):
        pass

    def sample(self, output_path, grid_type, resolution, contact):
        """Samples images using selected grid type."""

        # Set up output paths.
        img_path = os.path.join(output_path, "downloaded_images")
        sample_path = os.path.join(output_path, f"{grid_type}_samples")
        if not os.path.isdir(sample_path):
            os.mkdir(sample_path)

        # Loop through all images and sample desired z-slices on grid.
        for file_dir in glob(f"{img_path}/*/"):
            image_file = file_dir.replace(f"{img_path}/", "").replace("/", "")
            print(f"Sampling image {image_file} ...")
            sub_sample_path = os.path.join(sample_path, image_file)
            sub_image_path = os.path.join(img_path, image_file)

            # Create subdirectory for sample data for image file. Skip sampling
            # if a folder already exists for the image.
            if os.path.isdir(sub_sample_path):
                print(f"Skipping sampling for {image_file} ...")
                continue
            else:
                os.mkdir(sub_sample_path)

            # Get sampling coordinates.
            slices, offsets = self.get_image_slices(
                img_path, grid_type, image_file, resolution
            )
            sample_centers = self.get_sample_centers(
                grid_type, sub_image_path, resolution
            )

            # Sample all image slices.
            data = []
            for (z, slice), offset in zip(slices, offsets):
                slice_data = self.sample_image_slice(z, slice, offset, sample_centers)
                if slice_data is not None:
                    data = data + slice_data

            # Save data to csv.
            df = pd.DataFrame(data, columns=OUTPUT_COLUMNS)
            df.to_csv(
                os.path.join(sub_sample_path, f"{grid_type}_samples_{image_file}.csv"),
                index=False,
            )

            if contact:
                print("Saving contact sheet ...")
                self.save_contact_sheet(
                    df, os.path.join(sub_sample_path, f"contact_{image_file}.png")
                )

    @staticmethod
    def get_image_slices(output_path, grid_type, image_name, resolution):
        """Get list of z slices to sample at given resolution."""

        # Get number of z slice files.
        name_template = f"{output_path}/{image_name}/{image_name}_z%s"
        z_files = glob(name_template % "*")
        zmax = len(z_files)
        n_slices = 1 + int(math.ceil((zmax - 1) * SCALE_MICRONS_Z / resolution))

        # Create list of z slices with z and corresponding slice file name.
        z_list = np.arange(0, zmax - 1, math.floor((zmax - 1) / (n_slices - 1)))
        z_slices = [(z, name_template % str(z)) for z in z_list]

        # Create z offsets (only for hex to shift every other layer).
        if grid_type == "hex":
            z_offsets = [
                (i % 2) * ((resolution / SCALE_MICRONS) / 2)
                for i in range(len(z_slices))
            ]
        elif grid_type == "cartesian":
            z_offsets = [0] * len(z_slices)

        return z_slices, z_offsets

    @staticmethod
    def get_sample_centers(grid_type, image_path, resolution):
        """Get list of (x, y) samples for given grid type."""
        if grid_type == "hex":
            return SampleImages._get_hexagonal_centers(image_path, resolution)
        elif grid_type == "cartesian":
            return SampleImages._get_cartesian_centers(image_path, resolution)

    @staticmethod
    def _get_hexagonal_centers(image_path, resolution):
        """Get list of (x, y) samples for hexagonal grid using a sample 2D image slice."""
        sample_filename = os.listdir(image_path)[0]
        fov_image_slice = imageio.imread(os.path.join(image_path, sample_filename))
        pixel_resolution = math.ceil(resolution / SCALE_MICRONS)
        hex_centers, _ = create_hex_grid(
            nx=math.floor(fov_image_slice.shape[0] / pixel_resolution),
            ny=math.floor(fov_image_slice.shape[1] / pixel_resolution * math.sqrt(3)),
            min_diam=pixel_resolution,
            align_to_origin=False,
            do_plot=False,
        )
        return hex_centers

    @staticmethod
    def _get_cartesian_centers(image_path, resolution):
        """Get list of (x, y) samples for cartesian grid using a sample 2D image slice."""
        sample_filename = os.listdir(image_path)[0]
        fov_image_slice = imageio.imread(os.path.join(image_path, sample_filename))
        pixel_resolution = math.ceil(resolution / SCALE_MICRONS)
        x_pixels = pixel_resolution * np.arange(
            math.floor(fov_image_slice.shape[0] / pixel_resolution)
        )
        y_pixels = pixel_resolution * np.arange(
            math.floor(fov_image_slice.shape[1] / pixel_resolution)
        )
        return np.array(np.meshgrid(x_pixels, y_pixels)).T.reshape(-1, 2)

    @staticmethod
    def sample_image_slice(z, image_file, offset, sample_centers):
        """Sample image slice at given centers."""

        # Separate lists of x and y coordinates to sample and add offsets.
        xlist = [int(round(center[0] + offset)) for center in sample_centers[:]]
        ylist = [int(round(center[1] + offset)) for center in sample_centers[:]]

        # Load image file.
        fov_image_slice = imageio.imread(image_file + ".png")

        # Cycle through all x and y coordinates.
        result = []
        for x, y in zip(xlist, ylist):
            if y < fov_image_slice.shape[1] and x < fov_image_slice.shape[0]:
                v = fov_image_slice[x][y]
                if v > 0:
                    result.append([v, x, y, z])

        return result

    @staticmethod
    def save_contact_sheet(df, output_file, size=5):
        """Save contact sheet image for all z slices in dataframe."""
        max_id = int(df.id.max())
        min_id = int(df.id.min())
        cmap = cm.get_cmap("jet", max_id - min_id + 1)

        slices = sorted(df.z.unique())
        cols = math.ceil(np.sqrt(len(slices)))
        rows = math.ceil(len(slices) / cols)

        fig, ax = plt.subplots(
            rows, cols, figsize=(cols * size, rows * size), sharex="all", sharey="all"
        )
        subplots = [(row, col) for row in range(rows) for col in range(cols)]

        for i, (row, col) in enumerate(subplots):
            if i >= len(slices):
                ax[row, col].axis("off")
                continue

            x = df.x[df.z == slices[i]]
            y = df.y[df.z == slices[i]]
            ids = df.id[df.z == slices[i]]

            ax[row, col].scatter(x, y, s=10, c=ids, cmap=cmap, vmin=min_id, vmax=max_id)
            ax[row, col].set_title(f"z = {slices[i]:.3f}")
            ax[row, col].axis("off")

        fig.tight_layout(pad=0.1)
        fig.gca().invert_yaxis()
        plt.savefig(os.path.join(output_file))
