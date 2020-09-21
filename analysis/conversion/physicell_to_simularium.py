#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import numpy as np
from simulariumio import Converter

def main():
    parser = argparse.ArgumentParser(
        description="Converts PhysiCell trajectories to "
                    ".simularium JSON format for visualization"
    )
    parser.add_argument(
        "dir_path", help="the file path of the directory\
         containing the trajectories to parse")
    args = parser.parse_args()
    
    box_size = 1000.0
    scale = 0.01

    data = {
        "box_size" : np.array([box_size, box_size, 1.0 / scale]),
        "timestep" : 360.0,
        "path_to_output_dir" : args.dir_path,
        "types" : {
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
            }
        },
        "scale_factor" : scale,
    }
    Converter(data, "physicell").write_JSON("physicell_trajectory")

if __name__ == '__main__':
    main()
