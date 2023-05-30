#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from tarfile import TarFile
import tempfile
import json

import numpy as np

from simulariumio.physicell import (
    PhysicellConverter,
    PhysicellData,
)
from simulariumio import MetaData, UnitData, DisplayData, CameraData, DISPLAY_TYPE, JsonWriter
from simulariumio.filters import TranslateFilter


def convert_physicell_to_simularium(
    tar_file: TarFile, 
    box_size: list[float],
    timestep: float,
) -> str:
    working_dir = tempfile.TemporaryDirectory()
    tar_file.extractall(working_dir)
    box_size = np.array(box_size)
    scale = 0.1
    offset = scale * (-0.5 * box_size)
    data = PhysicellData(
        timestep=float(timestep),
        path_to_output_dir=working_dir,
        meta_data=MetaData(
            box_size=box_size,
            scale_factor=scale,
            camera_defaults=CameraData(
                position=np.array([30., -10., 30.]),
            ),
        ),
        display_data={
            0 : DisplayData(
                name="Substrate",
                display_type=DISPLAY_TYPE.SPHERE,
            )
        },
        time_units=UnitData("min"),
    )
    converter = PhysicellConverter(data)
    converter._data.spatial_units = UnitData(
        name="micron",
        magnitude=10.0,
    )
    colors = [
        "#fee34d",
        "#f7b232",
        "#bf5736",
        "#94a7fc",
        "#ce8ec9",
        "#58606c",
        "#0ba345",
        "#9267cb",
        "#81dbe6",
        "#bd7800",
        "#bbbb99",
        "#5b79f0",
        "#89a500",
        "#da8692",
        "#418463",
        "#9f516c",
        "#00aabf",
    ]
    type_names = []
    MAX_OWNER_CELLS = 10000  # from PhysiCell model
    for time_index in range(len(converter._data.agent_data.types)):
        for agent_index in range(len(converter._data.agent_data.types[time_index])):
            tn = converter._data.agent_data.types[time_index][agent_index]
            new_name = ""
            if "Substrate" in tn:
                new_name = "Substrate"  # hack to remove phase name
            if "cell" in tn:
                cell_id = int(tn[4:tn.index("#")])
                while cell_id > MAX_OWNER_CELLS:
                    cell_id -= MAX_OWNER_CELLS
                new_name = "Stem cell#" + str(cell_id)
                if new_name not in type_names:
                    type_names.append(new_name)
            if not new_name:
                continue
            converter._data.agent_data.types[time_index][agent_index] = new_name
    color_index = 0
    color_map = {}
    converter._data.agent_data.display_data = {
        "Substrate" : DisplayData(
            name="Substrate",
            display_type=DISPLAY_TYPE.SPHERE,
            color="#d0c5c8",
        )
    }
    for type_name in type_names:
        if type_name not in color_map:
            color_map[type_name] = colors[color_index]
            color_index += 1
            if color_index >= len(colors):
                color_index = 0
        converter._data.agent_data.display_data[type_name] = DisplayData(
            name=type_name,
            display_type=DISPLAY_TYPE.SPHERE,
            color=color_map[type_name],
        )
    converter._data.agent_data.n_agents[0] -= 1 # remove the huge cell artifact
    filtered_data = converter.filter_data([
        TranslateFilter(
            default_translation=offset,
        ),
    ])
    json_data = JsonWriter.format_trajectory_data(filtered_data)
    return json.dumps(json_data)
