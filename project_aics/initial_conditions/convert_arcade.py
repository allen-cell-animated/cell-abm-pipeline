import random
import io
from math import ceil, sqrt, pi
import xml.etree.ElementTree as ET

from project_aics.initial_conditions.__config__ import POTTS_TERMS
from project_aics.initial_conditions.process_samples import ProcessSamples
from project_aics.utilities.load import load_dataframe
from project_aics.utilities.save import save_json, save_buffer
from project_aics.utilities.keys import make_folder_key, make_file_key

CRITICAL_HEIGHT = 5


class ConvertARCADE:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "samples", "PROCESSED", False),
            "output": make_folder_key(context.name, "converted", "ARCADE", False),
        }
        self.files = {
            "input": make_file_key(context.name, ["PROCESSED", "csv"], "%s", ""),
            "cells": make_file_key(context.name, ["CELLS", "json"], "%s", ""),
            "locations": make_file_key(context.name, ["LOCATIONS", "json"], "%s", ""),
            "setup": make_file_key(context.name, ["xml"], "%s", ""),
        }

    def run(self, margin=[0, 0, 0]):
        for key in self.context.keys:
            self.convert_arcade(key, margin)

    def convert_arcade(self, key, margin):
        sample_key = self.folders["input"] + self.files["input"] % key
        samples_df = load_dataframe(self.context.working, sample_key)
        samples, bounds = self.extract_sample_voxels(samples_df, *margin)

        # Initialize outputs.
        locations, cells = [], []
        samples_by_id = samples.groupby("id")

        # Iterate through each cell in samples and convert.
        for i, (_, group) in enumerate(samples_by_id):
            cells.append(self.convert_to_cell(i + 1, group))
            locations.append(self.convert_to_location(i + 1, group))

        # Save converted ARCADE files.
        cells_key = self.folders["output"] + self.files["cells"] % key
        save_json(self.context.working, cells_key, cells)

        locations_key = self.folders["output"] + self.files["locations"] % key
        save_json(self.context.working, locations_key, locations)

        # Create and save setup file.
        setup = self.make_setup_file(len(cells), **bounds)
        setup_key = self.folders["output"] + self.files["setup"] % key
        save_buffer(self.context.working, setup_key, setup)

    @staticmethod
    def extract_sample_voxels(df, margin_x, margin_y, margin_z):
        """Load samples and reposition to center of bounding box."""

        # Get step size for voxels.
        step_x = ProcessSamples.get_step_size(df.x)
        step_y = ProcessSamples.get_step_size(df.y)
        step_z = ProcessSamples.get_step_size(df.z)

        # Rescale integers to step size 1.
        df["x"] = df["x"].divide(step_x).astype("int32")
        df["y"] = df["y"].divide(step_y).astype("int32")
        df["z"] = df["z"].divide(step_z).astype("int32")

        # Calculate voxel bounds
        bounds = {
            "length": df["x"].max() - df["x"].min() + 2 * margin_x + 3,
            "width": df["y"].max() - df["y"].min() + 2 * margin_y + 3,
            "height": df["z"].max() - df["z"].min() + 2 * margin_z + 3,
        }

        # Adjust bounds.
        df["x"] = df["x"] - df["x"].min() + margin_x + 1
        df["y"] = df["y"] - df["y"].min() + margin_y + 1
        df["z"] = df["z"] - df["z"].min() + margin_z + 1

        return df, bounds

    @staticmethod
    def make_setup_file(init, length, width, height, terms=POTTS_TERMS):
        """Create empty setup file for converted samples."""
        root = ET.fromstring("<set></set>")
        series = ET.SubElement(
            root,
            "series",
            {
                "name": "ARCADE",
                "interval": "1",
                "start": "0",
                "end": "0",
                "dt": "1",
                "ticks": "24",
                "ds": "1",
                "height": str(height),
                "length": str(length),
                "width": str(width),
            },
        )

        potts = ET.SubElement(series, "potts")
        for term in terms:
            ET.SubElement(potts, "potts.term", {"id": term})

        agents = ET.SubElement(series, "agents")
        populations = ET.SubElement(agents, "populations")
        ET.SubElement(populations, "population", {"id": "X", "init": str(init)})
        ET.indent(root, space="    ", level=0)
        return io.BytesIO(ET.tostring(root))

    @staticmethod
    def convert_to_cell(cell_id, samples):
        """Convert samples to ARCADE .CELLS json format."""
        volume, surface = ConvertARCADE.get_cell_targets(samples)
        state, phase = ConvertARCADE.get_cell_phase()

        return {
            "id": cell_id,
            "parent": 0,
            "pop": 1,
            "age": 0,
            "divisions": 0,
            "state": state,
            "phase": phase,
            "voxels": volume,
            "targets": [volume, surface],
        }

    @staticmethod
    def convert_to_location(cell_id, samples):
        """Convert samples to ARCADE .LOCATIONS json format."""
        center = ConvertARCADE.make_location_center(samples)
        voxels = ConvertARCADE.make_location_voxels(samples)

        return {
            "id": cell_id,
            "center": center,
            "location": [{"region": "UNDEFINED", "voxels": voxels}],
        }

    @staticmethod
    def get_cell_targets(samples):
        volume = len(samples)
        surface = ConvertARCADE.calculate_surface_area(volume, CRITICAL_HEIGHT)
        return volume, surface

    @staticmethod
    def get_cell_phase():
        if random.random() < 0.1:
            state = "APOPTOTIC"
            phase = "APOPTOTIC_EARLY"
        else:
            state = "PROLIFERATIVE"
            phase = "PROLIFERATIVE_G1"

        return state, phase

    @staticmethod
    def calculate_surface_area(volume, height):
        n = 0.50931200
        a = 0.79295247
        b = -1.54292969
        surface = 2 * volume / height + 2 * sqrt(pi) * sqrt(volume * height)
        correction = a * ((volume * height) ** n) + b
        return ceil(surface + correction)

    @staticmethod
    def make_location_center(samples):
        """Gets coordinates of center of samples."""
        return [int(samples[v].mean()) for v in ["x", "y", "z"]]

    @staticmethod
    def make_location_voxels(samples):
        """Get list of voxel coordinates from samples dataframe."""
        xyz_samples = samples[["x", "y", "z"]].to_records(index=False)
        return [[int(v) for v in voxel] for voxel in xyz_samples]
