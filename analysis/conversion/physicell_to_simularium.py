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
            0 : { # tumor cell
                0 : "tumor#G0G1",
                1 : "tumor#S",
                2 : "tumor#G2",
                3 : "tumor#M",
                4 : "tumor#apoptotic",
                5 : "tumor#necrotic",
            },
            1 : { # motile tumor cell
                0 : "motile tumor#G0G1",
                1 : "motile tumor#S",
                2 : "motile tumor#G2",
                3 : "motile tumor#M",
                4 : "motile tumor#apoptotic",
                5 : "motile tumor#necrotic",
            }
        },
        "scale_factor" : scale,
    }
    Converter(data, "physicell").write_JSON("physicell_trajectory")

if __name__ == '__main__':
    main()
