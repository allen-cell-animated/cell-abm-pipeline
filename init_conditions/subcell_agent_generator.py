#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import argparse
import imageio

import numpy as np
import pandas as pd
import quilt3
from matplotlib import pyplot as plt
from aicsimageio import AICSImage
from hexalattice.hexalattice import create_hex_grid


OUTPUT_COLUMNS = ["x", "y", "z", "id"]
SCALE_MICRONS = 0.108333  # from metadata.csv


class SubcellAgentGenerator:
    """
    Download and read AICS segmented cell fields and
    generate 3D positions and identities of initial subcell agents
    on a cartesian or hexagonal grid
    """
    @staticmethod
    def download_fov_images(output_path="", max_images=10):
        """
        Download 3D segmentation data for FOVs and save XY projections as PNGs
        """
        pkg = quilt3.Package.browse(
            "aics/hipsc_single_cell_image_dataset", "s3://allencell"
        )
        pkg_metadata = pkg["metadata.csv"]()
        metadata_path = os.path.join(output_path, "metadata.csv")
        if not os.path.isfile(metadata_path):
            with open(metadata_path, "w") as text_file:
                text_file.write(pkg_metadata.to_csv())
        fov_seg_paths = np.unique(
            pkg_metadata.sort_values(by=["FOVId"]).loc[:, "fov_seg_path"].values
        )
        np.random.shuffle(fov_seg_paths)
        root_path = os.path.join(output_path, "aics_images")
        if not os.path.isdir(root_path):
            os.mkdir(root_path)
        # for each segmented field of view
        for file_path in fov_seg_paths:
            max_images -= 1
            if max_images < 0:
                return
            file_name = os.path.basename(file_path)
            # find first '.' to remove '.ome' and '.tiff'
            file_name_noext = file_name[: file_name.find(".")]
            filedir = os.path.join(root_path, file_name_noext)
            if os.path.isdir(filedir):
                print(f"Skipping download for {file_name}...........")
                continue
            os.mkdir(filedir)
            # download data
            print(f"DOWNLOADING {file_name}")
            pkg["fov_seg_path"][file_name].fetch(
                filedir if filedir.endswith("/") else f"{filedir}/"
            )
            # save XY projection of segmented FOV as png
            img = AICSImage(os.path.join(filedir, file_name))
            for z in range(img.shape[3]):
                imageio.imwrite(
                    os.path.join(filedir, f"{file_name_noext}_z{z}.png"),
                    img.get_image_data("XYZ", S=0, T=0, C=1)[:, :, z],
                )

    @staticmethod
    def _get_hexagonal_centers(sub_img_path, resolution):
        """
        Get list of x and y values to sample on hexagonal grid using a sample 2D image slice
        """
        sample_filename = os.listdir(sub_img_path)[0]
        fov_image_slice = imageio.imread(os.path.join(sub_img_path, sample_filename))
        pixel_resolution = math.ceil(resolution / SCALE_MICRONS)
        hex_centers, _ = create_hex_grid(
            nx=math.floor(fov_image_slice.shape[0] / pixel_resolution),
            ny=math.floor(fov_image_slice.shape[1] / pixel_resolution),
            min_diam=pixel_resolution,
            align_to_origin=False,
            do_plot=False,
        )
        return hex_centers

    @staticmethod
    def _get_cartesian_centers(sub_img_path, resolution):
        """
        Get list of x and y values to sample on cartesian grid using a sample 2D image slice
        """
        sample_filename = os.listdir(sub_img_path)[0]
        fov_image_slice = imageio.imread(os.path.join(sub_img_path, sample_filename))
        pixel_resolution = math.ceil(resolution / SCALE_MICRONS)
        x_pixels = pixel_resolution * np.arange(
            math.floor(fov_image_slice.shape[0] / pixel_resolution) + 1
        )
        y_pixels = pixel_resolution * np.arange(
            math.floor(fov_image_slice.shape[1] / pixel_resolution) + 1
        )
        return np.array(np.meshgrid(x_pixels, y_pixels)).T.reshape(-1, 2)

    @staticmethod
    def _sample_slice_on_grid(
        use_hex_grid, file_path, zlist, sample_centers, sub_img_path, resolution=0
    ):
        """
        Sample 2D z slice on grid
        """
        # Only use the png files, which are the zslice images
        if not "png" in file_path:
            return None
        # Get z value
        zstart = file_path.rfind("z") + 1
        zend = file_path.rfind(".")
        z = int(file_path[zstart:zend])
        # check if this file is for a zslice in our list of z's to sample
        if z not in zlist:
            return None
        # Shift 2D hexgrid for every other z slice
        ind = np.where(zlist == z)[0][0]
        if use_hex_grid and ind % 2 == 1:
            pixel_offset = (resolution / SCALE_MICRONS) / 2
            xlist = [
                int(round(hex_center[0] + pixel_offset))
                for hex_center in sample_centers[:]
            ]
            ylist = [
                int(round(hex_center[1] + pixel_offset))
                for hex_center in sample_centers[:]
            ]
        else:
            xlist = [int(round(center[0])) for center in sample_centers[:]]
            ylist = [int(round(center[1])) for center in sample_centers[:]]
        # load image file
        file_name = os.path.basename(file_path)
        fov_image_slice = imageio.imread(os.path.join(sub_img_path, file_name))
        # get result dimensions
        n = 0
        for x, y in zip(xlist, ylist):
            if y < fov_image_slice.shape[1] and x < fov_image_slice.shape[0]:
                v = fov_image_slice[x][y]
                if v > 0:
                    n += 1
        if n < 1:
            return None
        result = np.zeros((int(n), 4))
        # cycle through all x and y coordinates to sample
        i = 0
        for x, y in zip(xlist, ylist):
            if y < fov_image_slice.shape[1] and x < fov_image_slice.shape[0]:
                v = fov_image_slice[x][y]
                if v > 0:
                    result[i][:] = [x, y, z, v]
                    i += 1
        return pd.DataFrame(result, columns=OUTPUT_COLUMNS)

    @staticmethod
    def sample_images_on_grid(use_hex_grid, output_path="", resolution=1.0):
        """
        Sample downloaded PNGs on a grid
        to get initial subcell agents in 3D
        """
        img_path = os.path.join(output_path, "aics_images")
        grid_type = "hex" if use_hex_grid else "cartesian"
        seed_path = os.path.join(output_path, f"{grid_type}_seeds")
        if not os.path.isdir(seed_path):
            os.mkdir(seed_path)
        # loop through all images, and sample desired z-slices on grid
        for file_dir in os.listdir(img_path):
            # Create subdirectory for seed data for this image file
            sub_seed_path = os.path.join(seed_path, file_dir)
            if os.path.isdir(sub_seed_path):
                print(f"Skipping sampling for {file_dir}...........")
                continue
            if not os.path.isdir(sub_seed_path):
                os.mkdir(sub_seed_path)
            # Get list of z values to sample
            sub_img_path = os.path.join(img_path, file_dir)
            zmax = len(os.listdir(sub_img_path)) - 1  # Subtract 1 for .tiff file
            n_slices = 1 + int(math.ceil((zmax - 1) * SCALE_MICRONS / resolution))
            zlist = np.arange(0, zmax - 1, math.floor((zmax - 1) / (n_slices - 1)))
            # Get list of x and y values to sample on grid using a sample 2D image slice
            if use_hex_grid:
                seed_centers = SubcellAgentGenerator._get_hexagonal_centers(
                    sub_img_path, resolution
                )
            else:
                seed_centers = SubcellAgentGenerator._get_cartesian_centers(
                    sub_img_path, resolution
                )
            # Compile coordinates and values into a dataframe and save as csv
            data = pd.DataFrame([], columns=OUTPUT_COLUMNS)
            for file_path in os.listdir(sub_img_path):
                slice_data = SubcellAgentGenerator._sample_slice_on_grid(
                    use_hex_grid,
                    file_path,
                    zlist,
                    seed_centers,
                    sub_img_path,
                    resolution,
                )
                if slice_data is not None:
                    data = data.append(slice_data)
            data.to_csv(
                os.path.join(sub_seed_path, f"{grid_type}_id_data_{file_dir}.csv"),
                index=False,
            )
            # Create and save scatterplots for each z-slice
            scatter_path = os.path.join(sub_seed_path, "scatter_by_z")
            if not os.path.isdir(scatter_path):
                os.mkdir(scatter_path)
            for z in zlist:
                if sum(data["z"] == z) > 0:
                    data_z = data.loc[data["z"] == z]
                    plt.clf()
                    plt.scatter(
                        data_z["x"],
                        data_z["y"],
                        c=data_z["id"],
                        cmap="jet",
                        s=5,
                        marker="o" if use_hex_grid else "s",
                    )
                    plt.axis("equal")
                    plt.savefig(
                        os.path.join(scatter_path, f"z{z}_{grid_type}_ids_{file_dir}"),
                        bbox_inches="tight",
                    )


def main():
    parser = argparse.ArgumentParser(
        description="Use AICS image data to create initial "
        "subcell agents for a simulation"
    )
    parser.add_argument(
        "n_images_download",
        nargs="?",
        help="how many new images to download?",
        default="10",
    )
    parser.add_argument(
        "resolution",
        nargs="?",
        help="how many microns between subcell agents?",
        default="1.0",
    )
    parser.add_argument(
        "grid_type",
        nargs="?",
        help="use a 'hex' or 'cartesian' grid to sample the images?",
        default="hex",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        help="path to aics_images directory, default to current directory",
        default="",
    )
    args = parser.parse_args()
    n_images_download = int(args.n_images_download)
    resolution = float(args.resolution)
    if n_images_download > 0:
        SubcellAgentGenerator.download_fov_images(args.output_path, n_images_download)
    SubcellAgentGenerator.sample_images_on_grid(
        "hex" in args.grid_type, args.output_path, resolution
    )


if __name__ == "__main__":
    main()
