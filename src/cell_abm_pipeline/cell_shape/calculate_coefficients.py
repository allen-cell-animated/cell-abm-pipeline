from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm
from aicsshparam import shparam, shtools

from cell_abm_pipeline.cell_shape.__config__ import COEFF_ORDER
from cell_abm_pipeline.utilities.load import load_tar, load_tar_member
from cell_abm_pipeline.utilities.save import save_dataframe
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key, make_full_key


class CalculateCoefficients:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "data.CELLS": make_folder_key(context.name, "data", "CELLS", False),
            "data.LOCATIONS": make_folder_key(context.name, "data", "LOCATIONS", False),
            "analysis": make_folder_key(context.name, "analysis", "SH", True),
        }
        self.files = {
            "data.CELLS": make_file_key(context.name, ["CELLS", "tar", "xz"], "%s", "%04d"),
            "data.LOCATIONS": make_file_key(context.name, ["LOCATIONS", "tar", "xz"], "%s", "%04d"),
            "analysis": lambda r: make_file_key(context.name, ["SH", r, "csv"], "%s", "%04d"),
        }

    def run(self, frames=[0], scale=1, region=None):
        for key in self.context.keys:
            for seed in self.context.seeds:
                self.calculate_coefficients(key, seed, frames, scale, region)

    def calculate_coefficients(self, key, seed, frames, scale, region):
        """
        Calculates spherical harmonics coefficients for all cells.
        """
        cell_data_key = make_full_key(self.folders, self.files, "data.CELLS", (key, seed))
        cell_data_tar = load_tar(self.context.working, cell_data_key)

        loc_data_key = make_full_key(self.folders, self.files, "data.LOCATIONS", (key, seed))
        loc_data_tar = load_tar(self.context.working, loc_data_key)

        all_coeffs = []
        member_names = [member.name.split(".")[0] for member in cell_data_tar.getmembers()]

        progress_bar = tqdm(frames)
        for member_name in member_names:
            frame = member_name.split("_")[-1]

            if int(frame) not in frames:
                continue

            cell_member = load_tar_member(cell_data_tar, f"{member_name}.CELLS.json")
            loc_member = load_tar_member(loc_data_tar, f"{member_name}.LOCATIONS.json")

            for cell, location in zip(cell_member, loc_member):
                voxels = self.get_location_voxels(location)
                array = self.make_voxels_array(voxels, scale)

                # Calculate alignment angle outside of get_shcoeffs so that angle can be used for
                # aligning regions (if needed).
                array, angle = shtools.align_image_2d(image=array)
                array = array.squeeze()

                if region:
                    region_voxels = self.get_location_voxels(location, region)
                    region_array = self.make_voxels_array(region_voxels, scale)
                    array = shtools.apply_image_alignment_2d(region_array, angle).squeeze()

                if np.sum(array) == 1:
                    continue

                (coeffs, _), _ = shparam.get_shcoeffs(
                    image=array, lmax=COEFF_ORDER, compute_lcc=False, alignment_2d=False
                )

                coeffs["KEY"] = key
                coeffs["ID"] = location["id"]
                coeffs["SEED"] = seed
                coeffs["TICK"] = frame
                coeffs["NUM_VOXELS"] = len(region_voxels) if region else len(voxels)
                coeffs["PHASE"] = cell["phase"]

                all_coeffs.append(coeffs)

            progress_bar.update()

        coeff_df = pd.DataFrame(all_coeffs)
        analysis_key = make_full_key(self.folders, self.files, "analysis", (key, seed), region)
        save_dataframe(self.context.working, analysis_key, coeff_df, index=False)

    @staticmethod
    def get_location_voxels(location, region=None):
        all_voxels = [
            voxel
            for loc in location["location"]
            for voxel in loc["voxels"]
            if not region or loc["region"] == region
        ]
        return all_voxels

    @staticmethod
    def scale_voxel_array(array, scale=1):
        array_scaled = array.repeat(scale, axis=0).repeat(scale, axis=1).repeat(scale, axis=2)
        return array_scaled

    @staticmethod
    def make_voxels_array(voxels, scale=1):
        """
        Converts list of voxels to array.
        """

        # Center voxels around (0,0,0).
        center_x, center_y, center_z = [round(x) for x in np.array(voxels).mean(axis=0)]
        all_xyz_centered = [(z - center_z, y - center_y, x - center_x) for x, y, z in voxels]

        # Create empty array.
        mins = np.min(all_xyz_centered, axis=0)
        maxs = np.max(all_xyz_centered, axis=0)
        height, width, length = np.subtract(maxs, mins) + 3
        array = np.zeros((height, width, length), dtype=np.uint8)

        # Fill in voxel array.
        all_xyz_offset = [
            (z - mins[0] + 1, y - mins[1] + 1, x - mins[2] + 1) for z, y, x in all_xyz_centered
        ]
        vals = [1] * len(all_xyz_offset)
        array[tuple(np.transpose(all_xyz_offset))] = vals

        # Scale the array if necessary.
        if scale != 1:
            array = CalculateCoefficients.scale_voxel_array(array, scale)

        return array

    @staticmethod
    def get_coeff_names(prefix: str = "", suffix: str = "", order: int = COEFF_ORDER) -> List[str]:
        """
        Get names of spherical harmonics coefficients.

        Parameters
        ----------
        prefix
            Prefix to prepend to each coefficient name
        suffix
            Suffix to append to each coefficient name
        order
            Order of the spherical harmonics parametrization
        """
        return [
            f"{prefix}shcoeffs_L{a}M{b}C{suffix}"
            for a in range(order + 1)
            for b in range(order + 1)
        ] + [
            f"{prefix}shcoeffs_L{a}M{b}S{suffix}"
            for a in range(order + 1)
            for b in range(order + 1)
        ]
