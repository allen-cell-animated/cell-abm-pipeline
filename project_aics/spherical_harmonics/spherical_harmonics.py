import numpy as np
import pandas as pd

from aicsshparam import shparam
from tqdm import tqdm

from ..utilities import *


class SphericalHarmonics:
    @staticmethod
    def run_calculate(name, keys, seeds, path, scale=1):
        for key in keys:
            for seed in seeds:
                SphericalHarmonics.calculate_coefficients(name, key, seed, path, scale)

    @staticmethod
    def calculate_coefficients(name, key, seed, path, scale):
        """
        Calculates spherical harmonics coefficients for all cells.
        """

        location_key = SphericalHarmonics.make_location_key(name, key, seed)
        location_tar = load_tar(path, location_key)

        all_coeffs = []

        for member_name in tqdm(location_tar.getmembers()):
            frame = member_name.name.split("_")[-1].replace(".LOCATIONS.json", "")
            member = load_tar_member(location_tar, member_name)

            for location in member:
                voxels = SphericalHarmonics.get_location_voxels(location)
                array = SphericalHarmonics.make_voxels_array(voxels)

                if scale != 1:
                    array = SphericalHarmonics.scale_voxel_array(voxels, scale)

                if np.sum(array) == 1:
                    continue

                (coeffs, _), _ = shparam.get_shcoeffs(image=array, lmax=16, compute_lcc=False)

                coeffs["KEY"] = key
                coeffs["ID"] = location["id"]
                coeffs["SEED"] = seed
                coeffs["TICK"] = frame

                all_coeffs.append(coeffs)

        df = pd.DataFrame(all_coeffs)
        df_key = SphericalHarmonics.make_output_key(name, key, seed)
        save_df(df, path, df_key, index=False)

    @staticmethod
    def make_location_key(name, key, seed):
        return f"{name}/data/data.LOCATIONS/{name}_{key}{seed:04d}.LOCATIONS.tar.xz"

    @staticmethod
    def make_output_key(name, key, seed):
        return f"{name}/analysis/{name}_{key}{seed:04d}.SH.csv"

    @staticmethod
    def get_location_voxels(location):
        all_voxels = []

        for region in location["location"]:
            all_voxels = all_voxels + region["voxels"]

        return all_voxels

    @staticmethod
    def scale_voxel_array(array, scale=1):
        array_scaled = array.repeat(scale, axis=0).repeat(scale, axis=1).repeat(scale, axis=2)
        return array_scaled

    @staticmethod
    def make_voxels_array(voxels):
        """
        Converts list of voxels to array.
        """

        # Center voxels around (0,0,0).
        x0, y0, z0 = [round(x) for x in np.array(voxels).mean(axis=0)]
        all_xyz_centered = [(z - z0, y - y0, x - x0) for x, y, z in voxels]

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

        return array
