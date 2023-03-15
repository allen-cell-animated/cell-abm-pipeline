from dataclasses import dataclass
from typing import Optional

import pandas as pd
from abm_shape_collection import get_shape_coefficients, make_voxels_array
from arcade_collection.output import extract_tick_json, get_location_voxels
from io_collection.keys import make_key
from io_collection.load import load_tar
from io_collection.save import save_dataframe
from prefect import flow

COEFFICIENT_ORDER = 16


@dataclass
class ParametersConfig:
    key: str

    seed: int

    scale: int

    frame: int

    region: Optional[str] = None

    order: int = COEFFICIENT_ORDER
    """Order of the spherical harmonics coefficient parametrization."""

    offset: int = 0

    chunk: Optional[int] = None


@dataclass
class ContextConfig:
    working_location: str


@dataclass
class SeriesConfig:
    name: str


@flow(name="calculate-coefficients")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    data_key = make_key(series.name, "data", "data.LOCATIONS")
    analysis_key = make_key(series.name, "analysis", "analysis.NEIGHBORS")
    series_key = f"{series.name}_{parameters.key}_{parameters.seed:04d}"

    locations_key = make_key(data_key, f"{series_key}.LOCATIONS.tar.xz")
    locations_tar = load_tar(context.working_location, locations_key)
    locations_json = extract_tick_json(locations_tar, series_key, parameters.frame, "LOCATIONS")

    all_coeffs = []

    count = 0

    for i, location in enumerate(locations_json):
        if i < parameters.offset:
            continue

        count = count + 1
        if parameters.chunk is not None and count > parameters.chunk:
            break

        voxels = get_location_voxels(location)

        if len(voxels) == 0:
            continue

        array = make_voxels_array(voxels, parameters.scale)

        if parameters.region is not None:
            region_voxels = get_location_voxels(location, parameters.region)

            if len(region_voxels) == 0:
                continue

            region_array = make_voxels_array(region_voxels, parameters.scale)
            coeffs = get_shape_coefficients(region_array, array, parameters.order)
        else:
            coeffs = get_shape_coefficients(array, array, parameters.order)

        coeffs["KEY"] = parameters.key
        coeffs["ID"] = location["id"]
        coeffs["SEED"] = parameters.seed
        coeffs["TICK"] = parameters.frame

        all_coeffs.append(coeffs)

    coeffs_dataframe = pd.DataFrame(all_coeffs)

    chunk_key = ""
    offset_key = f".{parameters.offset:04d}" if parameters.offset > 0 else ""

    if parameters.chunk is not None:
        chunk_key = f".{parameters.chunk:04d}"
        offset_key = f".{parameters.offset:04d}"

    region_key = f"_{parameters.region}" if parameters.region is not None else ""
    suffix = f"{region_key}{offset_key}{chunk_key}"

    coeffs_key = make_key(analysis_key, f"{series_key}_{parameters.frame:06d}{suffix}.COEFFS.csv")
    save_dataframe(context.working_location, coeffs_key, coeffs_dataframe, index=False)
