#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import os

import numpy as np
import pandas as pd
import quilt3

from lkaccess import LabKey
import lkaccess.contexts
from aicsimageio import AICSImage
from imageio import imwrite

def get_labkey_cell_data():
    '''
    download cell data from labkey if not already saved locally
    '''
    cell_data_path = Path("labkey_cells.pkl")
    if not cell_data_path.is_file():
        # Create our accessor class
        lk = LabKey(server_context=lkaccess.contexts.PROD)
        # Grab all production data
        cells = lk.dataset.get_pipeline_4_production_cells()
        # Put it into a dataframe
        result = pd.DataFrame(cells)
        result.to_pickle("labkey_cells.pkl")
        with open("cells.csv", "w") as text_file:
            text_file.write(result.to_csv())
        print("downloaded labkey data and saved as labkey_cells.pkl")
    else:
        result = pd.read_pickle("labkey_cells.pkl")
        print("loaded labkey data from local labkey_cells.pkl file")
    return result

def get_max_cells_fov_ids():
    '''
    group labkey cell data by FOVId and count number of cells per FOV
    '''
    labkey_cell_data = get_labkey_cell_data()
    seg_cell_counts = labkey_cell_data.groupby(["FOVId"]).size().sort_values(ascending=False)
    seg_cell_counts_csv = seg_cell_counts.to_csv()
    max_cells_fov_ids = []
    for i in range(10):
        max_cells_fov_ids.append(np.int64(seg_cell_counts.axes[0][i]))
    print(f"FOVs with most segmented cells: ids={max_cells_fov_ids}")
    with open("segmented_cell_counts.csv", "w") as text_file:
        text_file.write(seg_cell_counts_csv)
    print("saved count of segmented cells per FOV to segmented_cell_counts.csv")
    return max_cells_fov_ids

def download_max_cells_fov_images():
    '''
    download data for FOVs with most segmented cells and save XY projection as png
    '''
    max_cells_fov_ids = get_max_cells_fov_ids()
    pkg = quilt3.Package.browse("aics/hipsc_single_cell_image_dataset", "s3://allencell")
    pkg_metadata = pkg["metadata.csv"]()
    max_cells_fovs = pkg_metadata.loc[pkg_metadata["FOVId"].isin(max_cells_fov_ids)].sort_values(by=["FOVId"])
    file_paths = np.unique(max_cells_fovs.loc[:,"fov_seg_path"].values)
    with open("max_cells_fovs_metadata.csv", "w") as text_file:
        text_file.write(max_cells_fovs.to_csv())
    print("saved metadata for cells in FOVs with most segmented cells to quilt_metadata.csv")
    if not Path("aics_images").is_dir():
        os.mkdir("aics_images")
    for file_path in file_paths:
        file_name = Path(file_path).name
        if not Path(f"aics_images/{file_name}").is_file():
            # download data
            pkg["fov_seg_path"][file_name].fetch("aics_images/")
        if not Path(f"aics_images/{file_name}.png").is_file():
            # save XY projection of segmented FOV as png    
            img = AICSImage(f"aics_images/{file_name}")
            imwrite(f"aics_images/{file_name}.png", img.get_image_data("ZYX", S=0, T=0, C=1)[25])

def main():
    '''
    Save images of the fields of view with the most segmented cells
    '''
    download_max_cells_fov_images()

if __name__ == '__main__':
    main()