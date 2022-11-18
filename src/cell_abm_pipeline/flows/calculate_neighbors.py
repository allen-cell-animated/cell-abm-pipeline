from dataclasses import dataclass
from typing import Optional
from prefect import flow
import pandas as pd

from io_collection.keys import make_key
from io_collection.load import load_tar
from io_collection.save import save_dataframe
from arcade_collection.output import extract_tick_json


@dataclass
class ParametersConfig:
    key: str

    seed: int

    frame: int


@dataclass
class ContextConfig:
    working_location: str


@dataclass
class SeriesConfig:
    name: str


@flow(name="calculate-neighbors")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    series_key = f"{series.name}_{parameters.key}_{parameters.seed:04d}"

    locations_key = make_key(
        series.name, "data", "data.LOCATIONS", f"{series_key}.LOCATIONS.tar.xz"
    )
    locations_tar = load_tar(context.working_location, locations_key)
    locations_json = extract_tick_json(locations_tar, series_key, parameters.frame, "LOCATIONS")

    array = make_voxels_array(locations_json)

    neighbors_map = get_neighbors_map(array)
    depth_map = get_depth_map(array, neighbors_map)
    center_map = {location["id"]: location["center"] for location in locations_json}

    attributes = {"KEY": parameters.key, "SEED": parameters.seed, "TICK": parameters.frame}
    all_neighbors = flatten_neighbors_maps(neighbors_map, depth_map, center_map, attributes)

    neighbors_dataframe = pd.DataFrame(all_neighbors)
    neighbors_key = make_key(
        series.name,
        "analysis",
        "analysis.NEIGHBORS",
        f"{series_key}_{parameters.frame:06d}.NEIGHBORS.csv",
    )
    save_dataframe(context.working_location, neighbors_key, neighbors_dataframe, index=False)


from prefect import task
import numpy as np
from skimage import measure
from scipy import ndimage


@task
def flatten_neighbors_maps(neighbors_map, depth_map, center_map, attributes):
    neighbors = []

    for voxel_id, voxel_neighbors in neighbors_map.items():
        voxel_neighbors = {
            "ID": voxel_id,
            "GROUP": voxel_neighbors["group"],
            "NEIGHBORS": voxel_neighbors["neighbors"],
            "CX": center_map[voxel_id][0],
            "CY": center_map[voxel_id][1],
            "CZ": center_map[voxel_id][2],
            "DEPTH": depth_map[voxel_id],
        }
        voxel_neighbors.update(attributes)
        neighbors.append(voxel_neighbors)

    return neighbors


@task
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
    length, width, height = np.subtract(maxs, mins) + 3
    array = np.zeros((height, width, length), dtype=np.uint16)

    # Return if no voxels.
    if len(all_ids) == 0:
        return array

    # Fill voxel array.
    all_xyz_offset = [(z - mins[2] + 1, y - mins[1] + 1, x - mins[0] + 1) for x, y, z in all_xyz]
    array[tuple(np.transpose(all_xyz_offset))] = all_ids

    return array


@task
def get_neighbors_map(array):
    neighbors_map = {cell_id: {} for cell_id in np.unique(array)}
    neighbors_map.pop(0, None)

    # Create binary mask for array.
    mask = np.zeros(array.shape, dtype="int")
    mask[array != 0] = 1

    # Label connected groups.
    labels, groups = measure.label(mask, connectivity=2, return_num=True)

    # In line function that returns a filter lambda for a given id
    voxel_filter = lambda voxel_id: lambda v: voxel_id in v

    for group in range(1, groups + 1):
        group_crop = get_cropped_array(array, labels, group)
        voxel_ids = [i for i in np.unique(group_crop) if i != 0]

        # Find neighbors for each voxel id.
        for voxel_id in voxel_ids:
            voxel_crop = get_cropped_array(group_crop, voxel_id, crop_original=True)

            # Apply custom filter to get border locations.
            border_mask = ndimage.generic_filter(voxel_crop, voxel_filter(voxel_id), size=3)

            # Find neighbors overlapping border.
            neighbor_list = np.unique(voxel_crop[border_mask == 1])
            neighbor_list = [i for i in neighbor_list if i not in [0, voxel_id]]
            neighbors_map[voxel_id] = {"group": group, "neighbors": neighbor_list}

    return neighbors_map


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


def get_cropped_array(array, label, labels=None, crop_original=False):
    # Set all voxels not matching label to zero.
    array_mask = array.copy()
    labels = labels if labels else array_mask
    array_mask[labels != label] = 0

    # Crop array to label.
    xmin, xmax, ymin, ymax, zmin, zmax = get_bounding_box(array_mask)

    if crop_original:
        return array[xmin:xmax, ymin:ymax, zmin:zmax]

    return array_mask[xmin:xmax, ymin:ymax, zmin:zmax]


@task
def get_depth_map(array, neighbors_map):
    depth_map = {cell_id: 0 for cell_id in np.unique(array)}
    depth_map.pop(0, None)

    edge_ids = find_edge_ids(array)
    visited = set(edge_ids)
    queue = edge_ids.copy()

    while queue:
        current_id = queue.pop(0)

        current_neighbors = neighbors_map[current_id]["neighbors"]
        valid_neighbors = set(current_neighbors) - visited
        visited.update(valid_neighbors)
        queue = queue + list(valid_neighbors)

        for neighbor_id in valid_neighbors:
            depth_map[neighbor_id] = depth_map[current_id] + 1

        depth_map[current_id] = depth_map[current_id] + 1

    return depth_map


def find_edge_ids(array):
    slice_index = np.argmax(np.count_nonzero(array, axis=(1, 2)))
    array_slice = array[slice_index, :, :]

    # Calculate voronoi from cell shapes.
    distances = ndimage.distance_transform_edt(
        array_slice == 0, return_distances=False, return_indices=True
    )
    distances = distances.astype("uint16", copy=False)
    coordinates_y = distances[0].flatten()
    coordinates_x = distances[1].flatten()
    voronoi = array_slice[coordinates_y, coordinates_x].reshape(array_slice.shape)

    # Create border mask.
    mask = np.zeros(array_slice.shape, dtype="uint8")
    mask[array_slice != 0] = 1
    while measure.euler_number(mask) != 1:
        mask = ndimage.binary_dilation(mask, iterations=1)

    # Filter voronoi by mask to get edge ids.
    voronoi[mask == 1] = 0
    edge_ids = list(np.unique(voronoi))
    edge_ids.remove(0)

    return edge_ids
