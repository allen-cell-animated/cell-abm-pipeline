#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import numpy as np
from simulariumio.physicell import (
    PhysicellConverter, 
    PhysicellData,
)
from simulariumio import MetaData
from simulariumio.filters import TranslateFilter

def main():
    parser = argparse.ArgumentParser(
        description="Converts PhysiCell trajectories to "
                    ".simularium JSON format for visualization"
    )
    parser.add_argument(
        "dir_path", help="the file path of the PhysiCell\
        output directory containing the trajectory to parse")
    parser.add_argument(
        "output_name", help="filename for resulting \
        .simularium JSON file")
    args = parser.parse_args()

    box_size = np.array([980.0, 660.0, 160.0])
    scale = 0.01
    offset = scale * (-0.5 * box_size)

    data = PhysicellData(
        meta_data=MetaData(
            box_size=box_size,
            scale_factor=scale,
        ),
        timestep=36.0,
        path_to_output_dir=args.dir_path,
    )
    converter = PhysicellConverter(data)
    filtered_data = converter.filter_data([
        TranslateFilter(
            default_translation=offset
        )
    ])
    converter.write_external_JSON(filtered_data, args.output_name)

if __name__ == '__main__':
    main()
