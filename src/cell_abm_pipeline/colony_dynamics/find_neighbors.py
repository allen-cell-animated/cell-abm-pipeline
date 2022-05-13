import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import measure
from scipy import ndimage

from cell_abm_pipeline.utilities.load import load_tar, load_tar_member
from cell_abm_pipeline.utilities.save import save_dataframe
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key, make_full_key


class FindNeighbors:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "data.LOCATIONS": make_folder_key(context.name, "data", "LOCATIONS", False),
            "analysis": make_folder_key(context.name, "analysis", "NEIGHBORS", True),
        }
        self.files = {
            "data.LOCATIONS": make_file_key(context.name, ["LOCATIONS", "tar", "xz"], "%s", "%04d"),
            "analysis": make_file_key(context.name, ["NEIGHBORS", "csv"], "%s", "%04d"),
        }

    def run(self):
        for key in self.context.keys:
            for seed in self.context.seeds:
                self.find_neighbors(key, seed)

    def find_neighbors(self, key, seed):
        """
        Finds neighbor connections for all cells.
        """
        loc_data_key = make_full_key(self.folders, self.files, "data.LOCATIONS", (key, seed))
        loc_data_tar = load_tar(self.context.working, loc_data_key)

        all_neighbors = []
        member_names = [member.name.split(".")[0] for member in loc_data_tar.getmembers()]

        for member_name in tqdm(member_names):
            frame = member_name.split("_")[-1]
            member = load_tar_member(loc_data_tar, f"{member_name}.LOCATIONS.json")

            array = self.make_voxels_array(member)
            neighbors = self.get_array_neighbors(array)

            attributes = {"KEY": key, "SEED": seed, "TICK": frame}
            centers = {location["id"]: location["center"] for location in member}
            all_neighbors = all_neighbors + self.flatten_neighbors_list(
                neighbors, attributes, centers
            )

        neighbor_df = pd.DataFrame(all_neighbors)
        analysis_key = make_full_key(self.folders, self.files, "analysis", (key, seed))
        save_dataframe(self.context.working, analysis_key, neighbor_df, index=False)

    @staticmethod
    def flatten_neighbors_list(neighbors, attributes, centers):
        flattened_neighbors_list = []

        for group, voxel_id, neighbor_list in neighbors:
            center = centers[voxel_id]

            if len(neighbor_list) == 0:
                neighbor_list = [0]

            for neighbor in neighbor_list:
                entries = {"GROUP": group, "ID": voxel_id, "NEIGHBOR": neighbor}
                entries.update(attributes)
                entries.update({"CX": center[0], "CY": center[1], "CZ": center[2]})
                flattened_neighbors_list.append(entries)

        return flattened_neighbors_list

    @staticmethod
    def make_voxels_array(locations):
        # Extract all voxel positions with id.
        all_ids = []
        all_xyz = []
        for location in locations:
            cell_id = location["id"]
            xyz = [(x, y, z) for region in location["location"] for x, y, z in region["voxels"]]
            all_xyz = all_xyz + xyz
            all_ids = all_ids + [cell_id] * len(xyz)

        # Create empty array.
        mins = np.min(all_xyz, axis=0)
        maxs = np.max(all_xyz, axis=0)
        height, width, length = np.subtract(maxs, mins) + 3
        array = np.zeros((height, width, length), dtype=np.uint16)

        # Return if no voxels.
        if len(all_ids) == 0:
            return array

        # Fill voxel array.
        all_xyz_offset = [
            (z - mins[0] + 1, y - mins[1] + 1, x - mins[2] + 1) for z, y, x in all_xyz
        ]
        array[tuple(np.transpose(all_xyz_offset))] = all_ids

        return array

    @staticmethod
    def get_array_neighbors(array):
        """Get list of neighbors for each unique ID in array."""
        neighbors = []

        # Create binary mask for array.
        mask = np.zeros(array.shape, dtype="int")
        mask[array != 0] = 1

        # Label connected groups.
        labels, groups = measure.label(mask, connectivity=2, return_num=True)

        # In line function that returns a filter lambda for a given id
        voxel_filter = lambda voxel_id: lambda v: voxel_id in v

        for group in range(1, groups + 1):
            group_crop = FindNeighbors.get_cropped_array(array, labels, group)
            voxel_ids = [i for i in np.unique(group_crop) if i != 0]

            # Find neighbors for each voxel id.
            for voxel_id in voxel_ids:
                voxel_crop = FindNeighbors.get_cropped_array(
                    group_crop, voxel_id, crop_original=True
                )

                # Apply custom filter to get border locations.
                border_mask = ndimage.generic_filter(voxel_crop, voxel_filter(voxel_id), size=3)

                # Find neighbors overlapping border.
                neighbor_list = np.unique(voxel_crop[border_mask == 1])
                neighbor_list = [i for i in neighbor_list if i not in [0, voxel_id]]
                neighbors.append([group, voxel_id, neighbor_list])

        return neighbors

    @staticmethod
    def get_bounding_box(array):
        """Finds bounding box around binary array."""
        x, y, z = array.shape

        xbounds = np.any(array, axis=(1, 2))
        ybounds = np.any(array, axis=(0, 2))
        zbounds = np.any(array, axis=(0, 1))

        xmin, xmax = np.where(xbounds)[0][[0, -1]]
        ymin, ymax = np.where(ybounds)[0][[0, -1]]
        zmin, zmax = np.where(zbounds)[0][[0, -1]]

        xmin = max(xmin - 1, 0)
        xmax = min(xmax + 2, x)

        ymin = max(ymin - 1, 0)
        ymax = min(ymax + 2, y)

        zmin = max(zmin - 1, 0)
        zmax = min(zmax + 2, z)

        return xmin, xmax, ymin, ymax, zmin, zmax

    @staticmethod
    def get_cropped_array(array, label, labels=None, crop_original=False):
        # Set all voxels not matching label to zero.
        array_mask = array.copy()
        labels = labels if labels else array_mask
        array_mask[labels != label] = 0

        # Crop array to label.
        xmin, xmax, ymin, ymax, zmin, zmax = FindNeighbors.get_bounding_box(array_mask)

        if crop_original:
            return array[xmin:xmax, ymin:ymax, zmin:zmax]

        return array_mask[xmin:xmax, ymin:ymax, zmin:zmax]
