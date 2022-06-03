import io

import numpy as np
from simulariumio import (
    TrajectoryConverter,
    TrajectoryData,
    AgentData,
    UnitData,
    MetaData,
    DimensionData,
    ModelMetaData,
    CameraData,
    DisplayData,
)

from cell_abm_pipeline.convert_format.__config__ import PHASE_COLORS
from cell_abm_pipeline.utilities.load import load_tar, load_tar_member
from cell_abm_pipeline.utilities.save import save_buffer
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key, make_full_key


class ArcadeToSimularium:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input.CELLS": make_folder_key(context.name, "data", "CELLS", False),
            "input.LOCATIONS": make_folder_key(context.name, "data", "LOCATIONS", False),
            "output": make_folder_key(context.name, "converted", "SIMULARIUM", False),
        }
        self.files = {
            "input.CELLS": make_file_key(context.name, ["CELLS", "tar", "xz"], "%s", ""),
            "input.LOCATIONS": make_file_key(context.name, ["LOCATIONS", "tar", "xz"], "%s", ""),
            "output": make_file_key(context.name, ["simularium"], "%s", ""),
        }

    def run(self, ds=1, dt=1, box=(100, 100, 10)):
        for key in self.context.keys:
            self.arcade_to_simularium(key, ds, dt, box)

    def arcade_to_simularium(self, key, ds, dt, box):
        cell_data_key = make_full_key(self.folders, self.files, "input.CELLS", (key))
        cell_data_tar = load_tar(self.context.working, cell_data_key)

        loc_data_key = make_full_key(self.folders, self.files, "input.LOCATIONS", (key))
        loc_data_tar = load_tar(self.context.working, loc_data_key)

        frames = [member.name.split("_")[-1].split(".")[0] for member in cell_data_tar.getmembers()]
        length, width, height = box

        # Create agent data object.
        meta_data = self.get_meta_data(self.context.name, key, ds, *box)
        agent_data = self.get_agent_data(cell_data_tar, frames)
        agent_data.display_data = self.get_display_data(agent_data)

        # Iterate through each frame.
        for i, frame in enumerate(frames):
            prefix = f"{self.context.name}_{key}_{frame}"
            agent_data.times[i] = float(frame) * dt
            self.convert_cells_tar(agent_data, cell_data_tar, prefix, i)
            self.convert_locations_tar(agent_data, loc_data_tar, prefix, i, ds, *box)

        # Convert to Simularium format.
        data = TrajectoryConverter(
            TrajectoryData(
                meta_data=meta_data,
                agent_data=agent_data,
                time_units=UnitData("hr"),
                spatial_units=UnitData("um"),
            )
        ).to_JSON()

        output_key = make_full_key(self.folders, self.files, "output", (key,))
        with io.BytesIO() as buffer:
            buffer.write(data.encode("utf-8"))
            save_buffer(self.context.working, output_key, buffer)

    @staticmethod
    def get_meta_data(name, key, ds, length, width, height):
        meta_data = MetaData(
            box_size=np.array([length * ds, width * ds, height * ds]),
            camera_defaults=CameraData(
                position=np.array([10.0, 0.0, 200.0]),
                look_at_position=np.array([10.0, 0.0, 0.0]),
                fov_degrees=60.0,
            ),
            trajectory_title=f"ARCADE - {name} - {key}",
            model_meta_data=ModelMetaData(
                title="ARCADE",
                version="3.0",
                description=(f"Agent-based modeling framework ARCADE for {name} {key}."),
            ),
        )

        return meta_data

    @staticmethod
    def get_dimension_data(cells_tar, frames):
        total_frames = len(frames)

        max_agents = 0
        for member in cells_tar.getmembers():
            cells = load_tar_member(cells_tar, member)
            max_agents = max(max_agents, len(cells))

        return DimensionData(total_frames, max_agents)

    @staticmethod
    def get_agent_data(cells_tar, frames):
        dimension_data = ArcadeToSimularium.get_dimension_data(cells_tar, frames)
        return AgentData.from_dimensions(dimension_data)

    @staticmethod
    def get_display_data(data):
        display_data = {}
        for phase, color in PHASE_COLORS.items():
            display_data[phase] = DisplayData(name=phase, color=color)
        return display_data

    @staticmethod
    def convert_cells_tar(data, tar, prefix, index):
        member = load_tar_member(tar, f"{prefix}.CELLS.json")
        data.n_agents[index] = len(member)

        for i, cell in enumerate(member):
            data.unique_ids[index][i] = cell["id"]
            data.types[index].append(cell["phase"])
            data.radii[index][i] = (cell["voxels"] ** (1.0 / 3)) / 1.5

    @staticmethod
    def convert_locations_tar(data, tar, prefix, index, ds, length, width, height):
        member = load_tar_member(tar, f"{prefix}.LOCATIONS.json")

        for i, location in enumerate(member):
            data.positions[index][i] = np.array(
                [
                    (location["center"][0] - length / 2) * ds,
                    (location["center"][1] - width / 2) * ds,
                    (location["center"][2] - height / 2) * ds,
                ]
            )
