#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import numpy as np
from simulariumio.physicell import (
    PhysicellConverter, 
    PhysicellData,
)

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
    
    box_size = 1000.0

    data = PhysicellData(
        box_size=np.array([box_size, box_size, 1.0]),
        timestep=360.0,
        path_to_output_dir=args.dir_path,
        types={
            0 : {
                "name" : "tumor cell",
                4 : "A",
                10 : "B",
                12 : "C",
                13 : "D",
                100 : "E",
                101 : "F",
                102 : "G",
            },
            1 : {
                "name" : "motile tumor cell",
                4 : "A",
                10 : "B",
                12 : "C",
                13 : "D",
                100 : "E",
                101 : "F",
                102 : "G",
            },
        },
        scale_factor=0.01,
    )
    PhysicellConverter(data).write_JSON(args.output_name)

if __name__ == '__main__':
    main()
