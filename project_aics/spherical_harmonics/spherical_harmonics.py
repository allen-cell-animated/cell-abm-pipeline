import io
import lzma
import tarfile

import numpy as np
import pandas as pd
from tqdm import tqdm
from aicsshparam import shparam

from project_aics.utilities import load_tar, load_tar_member, load_buffer, save_df, save_buffer


class SphericalHarmonics:
    @staticmethod
    def run_calculate(name, keys, seeds, working, scale=1):
        for key in keys:
            for seed in seeds:
                SphericalHarmonics.calculate_coefficients(name, key, seed, working, scale)

    @staticmethod
    def run_compress(name, keys, seeds, working):
        for seed in seeds:
            SphericalHarmonics.compress_coefficients(name, keys, seed, working)

    @staticmethod
    def run_merge(name, keys, seeds, working):
        for seed in seeds:
            SphericalHarmonics.merge_coefficients(name, keys, seed, working)

    @staticmethod
    def calculate_coefficients(name, key, seed, working, scale):
        """
        Calculates spherical harmonics coefficients for all cells.
        """

        data_key = SphericalHarmonics.make_data_key(name, key, seed, "tar.xz", suffix=".LOCATIONS")
        data_tar = load_tar(working, data_key)

        all_coeffs = []

        for member_name in tqdm(data_tar.getmembers()):
            frame = member_name.name.split("_")[-1].replace(".LOCATIONS.json", "")
            member = load_tar_member(data_tar, member_name)

            for location in member:
                voxels = SphericalHarmonics.get_location_voxels(location)
                array = SphericalHarmonics.make_voxels_array(voxels)

                if scale != 1:
                    array = SphericalHarmonics.scale_voxel_array(array, scale)

                if np.sum(array) == 1:
                    continue

                (coeffs, _), _ = shparam.get_shcoeffs(image=array, lmax=16, compute_lcc=False)

                coeffs["KEY"] = key
                coeffs["ID"] = location["id"]
                coeffs["SEED"] = seed
                coeffs["TICK"] = frame

                all_coeffs.append(coeffs)

        coeff_df = pd.DataFrame(all_coeffs)
        analysis_key = SphericalHarmonics.make_analysis_key(name, seed, "SH.csv", key=key)
        save_df(working, analysis_key, coeff_df, index=False)

    @staticmethod
    def compress_coefficients(name, keys, seed, working):
        """
        Compress individual coefficients files into single archive.
        """
        file_keys = [
            SphericalHarmonics.make_analysis_key(name, seed, "SH.csv", key=key) for key in keys
        ]

        with io.BytesIO() as buffer:
            with tarfile.open(fileobj=buffer, mode="w:xz") as tar:
                for file_key in tqdm(file_keys):
                    contents = load_buffer(working, file_key)

                    info = tarfile.TarInfo(file_key.split("/")[-1])
                    info.size = contents.getbuffer().nbytes

                    tar.addfile(info, fileobj=contents)

            analysis_key = SphericalHarmonics.make_analysis_key(name, seed, "SH.tar.xz")
            save_buffer(working, analysis_key, buffer)

    @staticmethod
    def merge_coefficients(name, keys, seed, working):
        """
        Merge individual coefficients files into single file.
        """
        file_keys = [
            SphericalHarmonics.make_analysis_key(name, seed, "SH.csv", key=key) for key in keys
        ]

        with io.BytesIO() as buffer:
            with lzma.open(buffer, "wb") as lzf:
                for file_key in tqdm(file_keys):
                    contents = load_buffer(working, file_key)
                    file_contents = contents.read().decode("utf-8").split("\n")

                    if file_key == file_keys[0]:
                        header = file_contents[0] + "\n"
                        lzf.write(header.encode("utf-8"))

                    rows = [entry.replace("0.0,", "0,") for entry in file_contents[1:]]
                    lzf.write("\n".join(rows).encode("utf-8"))

            analysis_key = SphericalHarmonics.make_analysis_key(name, seed, "SH.csv.xz")
            save_buffer(working, analysis_key, buffer)

    @staticmethod
    def make_data_key(name, key, seed, extension, suffix=""):
        key = f"{key}_" if key else key
        return f"{name}/data/data{suffix}/{name}_{key}{seed:04d}{suffix}.{extension}"

    @staticmethod
    def make_analysis_key(name, seed, extension, key=""):
        key = f"{key}_" if key else key
        return f"{name}/analysis/{name}_{key}{seed:04d}.{extension}"

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

        return array