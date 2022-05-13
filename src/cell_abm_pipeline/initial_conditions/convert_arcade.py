import random
import io
from math import ceil, sqrt, pi
import xml.etree.ElementTree as ET

from cell_abm_pipeline.initial_conditions.__config__ import POTTS_TERMS
from cell_abm_pipeline.initial_conditions.process_samples import ProcessSamples
from cell_abm_pipeline.utilities.load import load_dataframe
from cell_abm_pipeline.utilities.save import save_json, save_buffer
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key, make_full_key

# Critical cell height (in voxels)
CRITICAL_HEIGHT = 5


class ConvertARCADE:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "samples", "PROCESSED", False),
            "cells": make_folder_key(context.name, "converted", "ARCADE", False),
            "locations": make_folder_key(context.name, "converted", "ARCADE", False),
            "setup": make_folder_key(context.name, "converted", "ARCADE", False),
        }
        self.files = {
            "input": lambda r: make_file_key(context.name, ["PROCESSED", r, "csv"], "%s", ""),
            "cells": make_file_key(context.name, ["CELLS", "json"], "%s", ""),
            "locations": make_file_key(context.name, ["LOCATIONS", "json"], "%s", ""),
            "setup": make_file_key(context.name, ["xml"], "%s", ""),
        }

    def run(self, margins=[0, 0, 0], region=None):
        for key in self.context.keys:
            self.convert_arcade(key, margins, region)

    def convert_arcade(self, key, margins, region):
        sample_key = make_full_key(self.folders, self.files, "input", key)
        samples_df = load_dataframe(self.context.working, sample_key)

        steps, offsets, bounds = self.calculate_sample_transform(samples_df, margins)
        samples = self.transform_sample_voxels(samples_df, steps, offsets)

        if region:
            region_key = make_full_key(self.folders, self.files, "input", key, region)
            region_df = load_dataframe(self.context.working, region_key)
            region_samples = self.transform_sample_voxels(region_df, steps, offsets)
            region_samples["region"] = region

            samples = samples.merge(region_samples, on=["id", "x", "y", "z"], how="left")
            samples["region"].fillna("DEFAULT", inplace=True)

        # Filter samples for valid samples.
        samples = self.filter_valid_samples(samples)

        # Initialize outputs.
        locations, cells = [], []
        samples_by_id = samples.groupby("id")

        # Iterate through each cell in samples and convert.
        for i, (_, group) in enumerate(samples_by_id):
            cells.append(self.convert_to_cell(i + 1, group))
            locations.append(self.convert_to_location(i + 1, group))

        # Save converted ARCADE files.
        cells_key = make_full_key(self.folders, self.files, "cells", key)
        save_json(self.context.working, cells_key, cells)

        locations_key = make_full_key(self.folders, self.files, "locations", key)
        save_json(self.context.working, locations_key, locations)

        # Create and save setup file.
        setup = self.make_setup_file(samples, **bounds)
        setup_key = make_full_key(self.folders, self.files, "setup", key)
        save_buffer(self.context.working, setup_key, setup)

    @staticmethod
    def calculate_sample_transform(df, margins):
        sizes = ["length", "width", "height"]
        coords = ["x", "y", "z"]

        steps = {i: ProcessSamples.get_step_size(df[i]) for i in coords}

        mins = {i: df[i].min() for i in coords}
        maxs = {i: df[i].max() for i in coords}

        offsets = {i: -(mins[i] / steps[i]) + margin + 1 for i, margin in zip(coords, margins)}

        bounds = {
            size: (maxs[i] - mins[i]) / steps[i] + 2 * margin + 3
            for i, margin, size in zip(coords, margins, sizes)
        }

        return steps, offsets, bounds

    @staticmethod
    def transform_sample_voxels(df, steps, offsets):
        """Scale samples and reposition to center of bounding box."""
        for i in ["x", "y", "z"]:
            df[i] = df[i].divide(steps[i]).astype("int32") + offsets[i]
        return df

    @staticmethod
    def make_setup_file(samples, length, width, height, terms=POTTS_TERMS):
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
                "height": str(int(height)),
                "length": str(int(length)),
                "width": str(int(width)),
            },
        )

        potts = ET.SubElement(series, "potts")
        for term in terms:
            ET.SubElement(potts, "potts.term", {"id": term})

        agents = ET.SubElement(series, "agents")
        populations = ET.SubElement(agents, "populations")

        init = len(samples.id.unique())
        population = ET.SubElement(populations, "population", {"id": "X", "init": str(init)})

        if "region" in samples.columns:
            for region in samples.region.unique():
                ET.SubElement(population, "population.region", {"id": region, "fraction": "0.5"})

        ET.indent(root, space="    ", level=0)
        return io.BytesIO(ET.tostring(root))

    @staticmethod
    def filter_valid_samples(samples):
        if "region" in samples.columns:
            # Ensure that each cell has all regions.
            samples = samples.groupby("id").filter(lambda x: len(x.region.unique()) > 1)

        return samples

    @staticmethod
    def convert_to_cell(cell_id, samples):
        """Convert samples to ARCADE .CELLS json format."""
        volume, surface = ConvertARCADE.get_cell_targets(samples)
        state, phase = ConvertARCADE.get_cell_phase(volume)

        cell = {
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

        if "region" in samples.columns:
            regions = ConvertARCADE.get_cell_regions(samples)
            cell.update({"regions": regions})

        return cell

    @staticmethod
    def convert_to_location(cell_id, samples):
        """Convert samples to ARCADE .LOCATIONS json format."""
        center = ConvertARCADE.make_location_center(samples)

        if "region" not in samples.columns:
            location = [
                {"region": "UNDEFINED", "voxels": ConvertARCADE.make_location_voxels(samples)}
            ]
        else:
            location = [
                {"region": region, "voxels": ConvertARCADE.make_location_voxels(group)}
                for region, group in samples.groupby("region")
            ]

        return {
            "id": cell_id,
            "center": center,
            "location": location,
        }

    @staticmethod
    def get_cell_targets(samples):
        volume = len(samples)
        surface = ConvertARCADE.calculate_surface_area(volume, CRITICAL_HEIGHT)
        return volume, surface

    @staticmethod
    def get_cell_phase(volume):
        thresholds = [250, 1000, 1124, 1726, 1969]
        states = [
            "APOPTOTIC",
            "APOPTOTIC",
            "PROLIFERATIVE",
            "PROLIFERATIVE",
            "PROLIFERATIVE",
        ]
        phases = ["LATE", "EARLY", "G1", "S", "G2"]

        index = next((ind for ind, thresh in enumerate(thresholds) if thresh > volume), -1)
        state = states[index]
        phase = f"{state}_{phases[index]}"

        return state, phase

    @staticmethod
    def get_cell_regions(samples):
        regions = []
        for region, group in samples.groupby("region"):
            regions.append(
                {
                    "region": region,
                    "voxels": len(group),
                    "targets": list(ConvertARCADE.get_cell_targets(group)),
                }
            )
        return regions

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
