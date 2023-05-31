import json
import tempfile
from tarfile import TarFile

import numpy as np
from simulariumio import DISPLAY_TYPE, CameraData, DisplayData, JsonWriter, MetaData, UnitData
from simulariumio.filters import TranslateFilter
from simulariumio.physicell import PhysicellConverter, PhysicellData


def convert_physicell_to_simularium(
    tar_file: TarFile,
    box_size: tuple[float, float, float],
    timestep: float,
    substrate_color: str,
    cell_colors: list[str],
    max_owner_cells: int = 10000,
) -> str:
    with tempfile.TemporaryDirectory() as temp_directory:
        # Extract contents of archive into temporary directory.
        tar_file.extractall(temp_directory)

        # Set sizing.
        box_size_array = np.array(box_size)
        scale = 0.1
        offset = scale * (-0.5 * box_size_array)

        # Create data object.
        data = PhysicellData(
            timestep=timestep,
            path_to_output_dir=temp_directory,
            meta_data=MetaData(
                box_size=box_size_array,
                scale_factor=scale,
                camera_defaults=CameraData(position=np.array([30.0, -10.0, 30.0])),
            ),
            display_data={
                0: DisplayData(
                    name="Substrate",
                    display_type=DISPLAY_TYPE.SPHERE,
                )
            },
            time_units=UnitData("min"),
        )

        # Create converter object.
        converter = PhysicellConverter(data)
        converter._data.spatial_units = UnitData(name="micron", magnitude=10.0)

    # Update agent names.
    type_names = []
    for time_index, _ in enumerate(converter._data.agent_data.types):
        for agent_index, _ in enumerate(converter._data.agent_data.types[time_index]):
            tn = converter._data.agent_data.types[time_index][agent_index]
            new_name = ""

            if "Substrate" in tn:
                new_name = "Substrate"

            if "cell" in tn:
                cell_id = int(tn[4 : tn.index("#")])

                while cell_id > max_owner_cells:
                    cell_id -= max_owner_cells

                new_name = "Stem cell#" + str(cell_id)

                if new_name not in type_names:
                    type_names.append(new_name)

            if not new_name:
                continue

            converter._data.agent_data.types[time_index][agent_index] = new_name

    # Set substrate agent color.
    converter._data.agent_data.display_data = {
        "Substrate": DisplayData(
            name="Substrate",
            display_type=DISPLAY_TYPE.SPHERE,
            color=substrate_color,
        )
    }

    # Set cell agent colors.
    color_map = {
        type_name: cell_colors[type_index % len(cell_colors)]
        for type_index, type_name in enumerate(type_names)
    }
    for type_name in type_names:
        converter._data.agent_data.display_data[type_name] = DisplayData(
            name=type_name,
            display_type=DISPLAY_TYPE.SPHERE,
            color=color_map[type_name],
        )

    # Remove the huge cell artifact.
    converter._data.agent_data.n_agents[0] -= 1

    # Center cells in box.
    filtered_data = converter.filter_data([TranslateFilter(default_translation=offset)])

    json_data = JsonWriter.format_trajectory_data(filtered_data)
    return json.dumps(json_data)
