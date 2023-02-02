from dataclasses import dataclass
from prefect import flow
import pandas as pd

from io_collection.keys import make_key
from io_collection.load import load_tar
from io_collection.save import save_dataframe
from arcade_collection.output import extract_tick_json
from abm_colony_collection.neighbor import make_voxels_array, get_neighbors_map, get_depth_map


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
    all_neighbors = []

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
        all_neighbors.append(voxel_neighbors)

    neighbors_dataframe = pd.DataFrame(all_neighbors)
    neighbors_key = make_key(
        series.name,
        "analysis",
        "analysis.NEIGHBORS",
        f"{series_key}_{parameters.frame:06d}.NEIGHBORS.csv",
    )
    save_dataframe(context.working_location, neighbors_key, neighbors_dataframe, index=False)