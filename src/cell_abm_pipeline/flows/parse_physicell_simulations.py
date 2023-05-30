from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from io_collection.load import load_tar
from io_collection.save import save_dataframe

from container_collection.manifest import filter_manifest_files
from io_collection.load import load_dataframe
from prefect import flow

from simulariumio.physicell.dep.pyMCDS import pyMCDS

MAX_OWNER_CELLS = 10000  # from PhysiCell model


@dataclass
class ParametersConfig:
    include_filters: list[str] = field(default_factory=lambda: ["*"])

    exclude_filters: list[str] = field(default_factory=lambda: [])


@dataclass
class ContextConfig:
    working_location: str

    manifest_location: str


@dataclass
class SeriesConfig:
    name: str

    manifest_key: str

    extensions: list[str]


def _radius_for_volume(total_volume: float) -> float:
    return np.cbrt(3.0 / 4.0 * total_volume / np.pi)


@flow(name="parse-physicell-simulations")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    manifest = load_dataframe(context.manifest_location, series.manifest_key)
    filtered_files = filter_manifest_files(
        manifest, series.extensions, parameters.include_filters, parameters.exclude_filters
    )

    for key, files in filtered_files.items():
        # TODO verify this will load tar
        tar_file = load_tar(context.working_location, key) 
        working_dir = context.working_location
        # load data and shape for analysis
        output_files = working_dir.glob("*output*.xml")
        file_mapping = {}
        for output_file in output_files:
            index = int(output_file.name[output_file.name.index("output") + 6 :].split(".")[0])
            file_mapping[index] = output_file
        physicell_data = []
        for _, xml_file in sorted(file_mapping.items()):
            physicell_data.append(pyMCDS(xml_file.name, False, working_dir).get_cell_df())
            
        # this might be all you need for analysis
        print(physicell_data)
        
        # or do some more shaping:
        result = {
            "time_ix": [],
            "owner_id": [], 
            "pos_x": [], 
            "pos_y": [], 
            "pos_z": [], 
            "radius": [],
        }
        for time_ix in range(len(physicell_data)):
            physicell_cells = pyMCDS(xml_file.name, False, working_dir).get_cell_df()
            n_subcells = int(len(physicell_cells["cell_type"]))
            for cell_index in range(n_subcells):
                subcell_type_id = int(physicell_cells["cell_type"][cell_index])
                owner_cell_id = subcell_type_id
                while owner_cell_id - MAX_OWNER_CELLS > 0:
                    owner_cell_id -= MAX_OWNER_CELLS
                result["time_ix"].append(time_ix)
                result["owner_id"].append(owner_cell_id)
                result["pos_x"].append(physicell_cells["position_x"][cell_index])
                result["pos_y"].append(physicell_cells["position_y"][cell_index])
                result["pos_z"].append(physicell_cells["position_z"][cell_index])
                result["radius"].append(_radius_for_volume(
                    physicell_cells["total_volume"][cell_index]
                ))
        result_df = pd.DataFrame.from_dict(result)
        spatial_units = physicell_data[0].data["metadata"]["spatial_units"]
        print(result_df)
        print(spatial_units)
        
        # TODO shape data for analysis and save
        save_dataframe(context.working_location, key, result_df, index=False)
