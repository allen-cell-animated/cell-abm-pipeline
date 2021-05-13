#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import imageio
import argparse

import numpy as np
import pandas as pd
import quilt3
from matplotlib import pyplot as plt
from aicsimageio import AICSImage
from hexalattice.hexalattice import create_hex_grid


OUTPUT_COLUMNS = ["x", "y", "z", "id"]

class SubcellAgentGenerator:
    """
    Download and read AICS segmented cell fields and 
    generate 3D positions and identities of initial subcell agents
    """
    @staticmethod
    def download_fov_images(path_to_aics_images="", max_images=10):
        """
        Download data for FOVs and save XY projections as PNGs
        """
        pkg = quilt3.Package.browse(
            "aics/hipsc_single_cell_image_dataset", "s3://allencell"
        )
        pkg_metadata = pkg["metadata.csv"]()
        fov_seg_paths = (
            np.unique(pkg_metadata.sort_values(by=["FOVId"]).loc[:, "fov_seg_path"].values)
        )
        np.random.shuffle(fov_seg_paths)
        root_path = os.path.join(path_to_aics_images, "aics_images")
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
            pkg["fov_seg_path"][file_name].fetch(filedir if filedir.endswith("/") else f"{filedir}/")
            # save XY projection of segmented FOV as png
            img = AICSImage(os.path.join(filedir, file_name))
            for z in range(img.shape[3]):
                imageio.imwrite(
                    os.path.join(filedir, f"{file_name_noext}_z{z}.png"),
                    img.get_image_data("XYZ", S=0, T=0, C=1)[:, :, z],
                )

    @staticmethod
    def _sample_slice_hex_grid(file_path, zlist, hex_centers, resolution, sub_img_path):
        """
        Sample 2D hexgrid on each z slice
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
        if ind % 2 == 0:
            xlist = [int(round(hex_center[0])) for hex_center in hex_centers[:]]
            ylist = [int(round(hex_center[1])) for hex_center in hex_centers[:]]
        else:
            xlist = [
                int(round(hex_center[0] + (resolution / 2)))
                for hex_center in hex_centers[:]
            ]
            ylist = [
                int(round(hex_center[1] + (resolution / 2)))
                for hex_center in hex_centers[:]
            ]
        # load image file
        file_name = os.path.basename(file_path)
        fov_image_slice = imageio.imread(os.path.join(sub_img_path, file_name))
        # cycle through all x and y coordinates to sample
        result = pd.DataFrame([], columns=OUTPUT_COLUMNS)
        for x, y in zip(xlist, ylist):
            if y < fov_image_slice.shape[1] and x < fov_image_slice.shape[0]:
                v = fov_image_slice[x][y]
                if v > 0:
                    result = result.append({"x" : x, "y" : y, "z" : z, "id": v}, ignore_index=True)
        return result

    @staticmethod
    def sample_images_hex_grid(path_to_aics_images="", resolution=10):
        """
        Sample downloaded PNGs on a hexagonal grid
        to get initial subcell agents in 3D
        """
        img_path = os.path.join(path_to_aics_images, "aics_images")
        hex_path = os.path.join(path_to_aics_images, "hex_seeds")
        if not os.path.isdir(hex_path):
            os.mkdir(hex_path)
        # loop through all images, and sample desired z-slices on hexgrid
        for file_dir in os.listdir(img_path):
            # Create subdirectory for hex seed data for this image file
            sub_hex_path = os.path.join(hex_path, file_dir)
            if os.path.isdir(sub_hex_path):
                print(f"Skipping sampling for {file_dir}...........")
                continue
            if not os.path.isdir(sub_hex_path):
                os.mkdir(sub_hex_path)
            # Get list of z values to sample
            sub_img_path = os.path.join(img_path, file_dir)
            zmax = (
                len(os.listdir(sub_img_path)) - 2
            )  # Subtract 2: 1 for 0-indexing and 1 for .tiff file
            zlist = np.arange(0, zmax, resolution)
            # Get list of x and y values to sample on hexgrid using a sample 2D image slice
            sample_filename = os.listdir(sub_img_path)[0]
            fov_image_slice = imageio.imread(os.path.join(sub_img_path, sample_filename))
            hex_centers, _ = create_hex_grid(
                nx=round(fov_image_slice.shape[0] / resolution),
                ny=round(fov_image_slice.shape[1] / resolution),
                min_diam=resolution,
                align_to_origin=False,
                do_plot=False,
            )
            # Compile coordinates and values into a dataframe and save as csv
            data = pd.DataFrame([], columns=OUTPUT_COLUMNS)
            for file_path in os.listdir(sub_img_path):
                slice_data = SubcellAgentGenerator._sample_slice_hex_grid(
                    file_path, zlist, hex_centers, resolution, sub_img_path
                )
                if slice_data is not None:
                    data = data.append(slice_data)
            data.to_csv(
                os.path.join(sub_hex_path, f"hex_id_data_{file_dir}.csv"), index=False
            )
            # Create and save scatterplots for each z-slice
            hex_scatter_path = os.path.join(sub_hex_path, "scatter_by_z")
            if not os.path.isdir(hex_scatter_path):
                os.mkdir(hex_scatter_path)
            for z in zlist:
                if sum(data["z"] == z) > 0:
                    data_z = data.loc[data["z"] == z]
                    plt.clf()
                    plt.scatter(data_z["x"], data_z["y"], c=data_z["id"], cmap="jet", s=5)
                    plt.axis("equal")
                    plt.savefig(
                        os.path.join(hex_scatter_path, f"z{z}_hex_ids_{file_dir}"),
                        bbox_inches="tight",
                    )

    @staticmethod
    def sample_images_cartesian_grid(path_to_aics_images="", resolution=10):
        """
        Sample downloaded PNGs on a cartesian grid
        to get initial subcell agents in 3D
        """
        # TODO
        print("Cartesian sampling still TODO")

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
        "grid_type",
        nargs="?",
        help="use a 'hexagonal' or 'cartesian' grid to sample the images?",
        default="hexagonal",
    )
    parser.add_argument(
        "aics_images_path",
        nargs="?",
        help="path to aics_images directory, default to current directory",
        default="",
    )
    parser.add_argument(
        "resolution",
        nargs="?",
        help="how many pixels per subcell agent?",
        default="10",
    )
    args = parser.parse_args()
    n_images_download = int(args.n_images_download)
    resolution = int(args.resolution)
    if n_images_download > 0:
        SubcellAgentGenerator.download_fov_images(args.aics_images_path, n_images_download)
    if "hexagonal" in args.grid_type:
        SubcellAgentGenerator.sample_images_hex_grid(args.aics_images_path, resolution)
    else:
        SubcellAgentGenerator.sample_images_cartesian_grid(args.aics_images_path, resolution)


if __name__ == "__main__":
    main()
