#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import os
import math
import imageio

import numpy as np
import pandas as pd
import quilt3
from matplotlib import pyplot as plt
from aicsimageio import AICSImage
from hexalattice.hexalattice import *


def download_fov_images(max_images=10):
    '''
    download data for FOVs and save XY projection as png
    '''
    pkg = quilt3.Package.browse("aics/hipsc_single_cell_image_dataset", "s3://allencell")
    pkg_metadata = pkg["metadata.csv"]()
    fov_seg_paths = np.unique(pkg_metadata.sort_values(by=["FOVId"]).loc[:,"fov_seg_path"].values)
    if not Path("aics_images").is_dir():
        os.mkdir("aics_images")
    for file_path in fov_seg_paths:
        max_images -= 1
        if max_images < 0:
            return
        file_name = Path(file_path).name
        endname_ind = file_name.find(".")
        file_name_noext = file_name[:endname_ind]
        filedir = f"aics_images/{file_name_noext}/"
        if Path(filedir).is_dir():
            print(f"Skipping download for {file_name}")
            continue
        os.mkdir(filedir)
        # download data
        print(f"DOWNLOADING {file_name}")
        pkg["fov_seg_path"][file_name].fetch(filedir)
        # save XY projection of segmented FOV as png
        img = AICSImage(Path(f"{filedir}{file_name}"))
        for z in range(img.shape[3]):
            imageio.imwrite(f"{filedir}{file_name_noext}_z{z}.png", img.get_image_data("XYZ", S=0, T=0, C=1)[:,:,z])
                
def sample_hex_grid(resolution = 10):

    if not Path(f"hex_seeds").is_dir():
        os.mkdir("hex_seeds")

    # loop through all images, and sample desired zlices on hexgrid
    for file_dir in os.listdir("aics_images/"):

        # Create subdirectory for hex seed data for this image file
        if not Path(f"hex_seeds/{file_dir}").is_dir():
            os.mkdir(f"hex_seeds/{file_dir}")

        # Get list of z values to sample
        zmax = len(os.listdir(f"aics_images/{file_dir}")) - 2 # Subtract 2: 1 for 0-indexing and 1 for .tiff file
        zlist = np.arange(0, zmax, resolution)

        # Get list of x and y values to sample on hexgrid using a sample 2D image slice
        sample_filename = os.listdir(f"aics_images/{file_dir}")[0]
        fov_image_slice = imageio.imread(f"aics_images/{file_dir}/{sample_filename}")
        hex_centers, _ = create_hex_grid(
                nx=round(fov_image_slice.shape[0] / resolution),
                ny=round(fov_image_slice.shape[1] / resolution),
                min_diam=resolution,
                align_to_origin=False,
                do_plot=False,
            )

        # Initialize lists of coordinates and values to fill
        x_array = []
        y_array = []
        z_array = []
        values = []

        # Loop through files and sample 2D hexgrid on each z slice
        for file_path in os.listdir(f"aics_images/{file_dir}"):

            # Only use the png files which are the zslice images
            if "png" in file_path:

                # Get z value
                zstart = file_path.rfind("z")+1
                zend = file_path.rfind(".")
                z = int(file_path[zstart:zend])

                # check if this file is for a zslice in our list of z's to sample
                if z in zlist:

                    # Shift 2D hexgrid for every other z slice
                    ind = np.where(zlist == z)[0][0]
                    if ind % 2 == 0:
                        xlist = [int(round(hex_center[0])) for hex_center in hex_centers[:]]
                        ylist = [int(round(hex_center[1])) for hex_center in hex_centers[:]]

                    else:
                        xlist = [int(round(hex_center[0] + (resolution/2))) for hex_center in hex_centers[:]]
                        ylist = [int(round(hex_center[1] + (resolution/2))) for hex_center in hex_centers[:]]

                    # load image file
                    file_name = Path(file_path).name
                    fov_image_slice = imageio.imread(f"aics_images/{file_dir}/{file_name}")

                    # cycle through all x and y coordinates to sample
                    for x, y in zip(xlist, ylist):
                        if  y < fov_image_slice.shape[1] and x < fov_image_slice.shape[0]:
                            v = fov_image_slice[x][y]
                            if v > 0:
                                x_array.append(x)
                                y_array.append(y)
                                z_array.append(z)
                                values.append(v)

        # Compile coordinates and values into a dataframe and save as csv
        data = pd.DataFrame({"x": x_array, "y": y_array, "z": z_array, "id": values})
        data.to_csv(f"hex_seeds/{file_dir}/hex_id_data_{file_dir}.csv", index=False)

        if not Path(f"hex_seeds/{file_dir}/scatter_by_z").is_dir():
            os.mkdir(f"hex_seeds/{file_dir}/scatter_by_z")
        for z in zlist:
            if sum(data["z"] == z) > 0:
                data_z = data.loc[data["z"] == z]
                plt.clf()
                plt.scatter(data_z["x"], data_z["y"], c=data_z["id"], cmap="jet", s=5)
                plt.axis("equal")
                plt.savefig(f"hex_seeds/{file_dir}/scatter_by_z/z{z}_hex_ids_{file_name}",bbox_inches="tight")

def main():
    '''
    Save images of the fields of view with the most segmented cells
    '''
    download_fov_images()
    # sample_hex_grid()

if __name__ == '__main__':
    main()
