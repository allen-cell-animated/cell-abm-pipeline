from dataclasses import dataclass
from typing import Optional
from prefect import flow
import pandas as pd

from io_collection.keys import make_key
from io_collection.load import load_tar
from io_collection.save import save_dataframe
from arcade_collection.output import extract_tick_json, get_location_voxels
from cell_abm_pipeline.flows.tasks import make_voxels_array, get_spherical_harmonic_coefficients


@dataclass
class ParametersConfig:
    key: str

    seed: int

    scale: int

    frame: int

    region: Optional[str] = None

    lmax: int = 16


@dataclass
class ContextConfig:
    working_location: str


@dataclass
class SeriesConfig:
    name: str


@flow(name="calculate-coefficients")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    series_key = f"{series.name}_{parameters.key}_{parameters.seed:04d}"

    locations_key = make_key(
        series.name, "data", "data.LOCATIONS", f"{series_key}.LOCATIONS.tar.xz"
    )
    locations_tar = load_tar(context.working_location, locations_key)
    locations_json = extract_tick_json(locations_tar, series_key, parameters.frame, "LOCATIONS")

    all_coeffs = []

    for location in locations_json:
        voxels = get_location_voxels(location)

        if len(voxels) == 0:
            continue

        array = make_voxels_array(voxels, parameters.scale)

        if parameters.region is not None:
            region_voxels = get_location_voxels(location, parameters.region)
            region_array = make_voxels_array(region_voxels, parameters.scale)
            coeffs = get_spherical_harmonic_coefficients(region_array, array, parameters.lmax)
        else:
            coeffs = get_spherical_harmonic_coefficients(array, array, parameters.lmax)

        coeffs["KEY"] = parameters.key
        coeffs["ID"] = location["id"]
        coeffs["SEED"] = parameters.seed
        coeffs["TICK"] = parameters.frame

        all_coeffs.append(coeffs)

    coeffs_dataframe = pd.DataFrame(all_coeffs)
    region_key = f"_{parameters.region}" if parameters.region is not None else ""
    coeffs_key = make_key(
        series.name,
        "analysis",
        "analysis.COEFFS",
        f"{series_key}_{parameters.frame:06d}{region_key}.COEFFS.csv",
    )
    save_dataframe(context.working_location, coeffs_key, coeffs_dataframe, index=False)
