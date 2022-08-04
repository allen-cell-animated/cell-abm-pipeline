import io
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from cell_abm_pipeline.initial_conditions.__config__ import (
    POTTS_TERMS,
    VOLUME_AVGS,
    VOLUME_STDS,
    CRITICAL_VOLUME_AVGS,
    CRITICAL_VOLUME_STDS,
    HEIGHT_AVGS,
    HEIGHT_STDS,
    CRITICAL_HEIGHT_AVGS,
    CRITICAL_HEIGHT_STDS,
    CELL_STATE_THRESHOLD_FRACTIONS,
)
from cell_abm_pipeline.initial_conditions.__main__ import Context
from cell_abm_pipeline.initial_conditions.process_samples import ProcessSamples
from cell_abm_pipeline.utilities.load import load_dataframe
from cell_abm_pipeline.utilities.save import save_json, save_buffer
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key, make_full_key


class ConvertARCADE:
    """
    Task to convert samples into ARCADE input formats.

    Working location structure for a given context:

    .. code-block:: bash

        (name)
        ├── converted
        │    └── converted.ARCADE
        │        ├── (name)_(image key 1).xml
        │        ├── (name)_(image key 1).CELLS.json
        │        ├── (name)_(image key 1).LOCATIONS.json
        │        ├── (name)_(image key 2).xml
        │        ├── (name)_(image key 2).CELLS.json
        │        ├── (name)_(image key 2).LOCATIONS.json
        │        ├── ...
        │        ├── (name)_(image key n).xml
        │        ├── (name)_(image key n).CELLS.json
        │        └── (name)_(image key n).LOCATIONS.json
        └── samples
            └── samples.PROCESSED
                ├── (name)_(image key 1).PROCESSED.csv
                ├── (name)_(image key 2).PROCESSED.csv
                ├── ...
                ├── (name)_(image key n).PROCESSED.csv
                ├── ...
                ├── (name)_(image key 1).PROCESSED.(region).csv
                ├── (name)_(image key 2).PROCESSED.(region).csv
                ├── ...
                └── (name)_(image key n).PROCESSED.(region).csv

    For each image key, the conversion creates three ARCADE output files:
    **.xml**, **.CELLS.json**, and **.LOCATIONS.json**.
    For conversions with region, sample for the region should include the region
    key in the extension.

    Attributes
    ----------
    context
        **Context** object defining working location and name.
    folders
        Dictionary of input and output folder keys.
    files
        Dictionary of input and output file keys.
    """

    def __init__(self, context: Context):
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

    def run(
        self,
        margins: Tuple[int, int, int] = (0, 0, 0),
        region: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> None:
        """
        Runs convert ARCADE task for given context.

        Parameters
        ----------
        margins
            Margin size in x, y, and z directions.
        region
            Region key to include in conversion.
        reference
            Path to reference data for conversion.
        """
        if reference:
            reference_data = load_dataframe(self.context.working, reference)
        else:
            reference_data = pd.DataFrame([], columns=["key", "id"])

        for key in self.context.keys:
            key_reference_data = reference_data[reference_data.key == key]
            self.convert_arcade(key, margins, region, key_reference_data)

    def convert_arcade(
        self,
        key: str,
        margins: Tuple[int, int, int],
        region: Optional[str],
        reference: pd.DataFrame,
    ) -> None:
        """
        Convert ARCADE task.

        Loads processed samples from working location (along with processed
        region samples, if included).
        Iterates through each cell id in the samples and converts in the ARCADE
        formats for .CELLS and .LOCATIONS file formats.

        Parameters
        ----------
        key
            Key for samples.
        margins
            Margin size in x, y, and z directions.
        region
            Region key to include in conversion.
        reference
            Reference data for conversion.
        """
        sample_key = make_full_key(self.folders, self.files, "input", key)
        processed_samples = load_dataframe(self.context.working, sample_key)
        samples = self.transform_sample_voxels(processed_samples, margins)

        if region:
            region_key = make_full_key(self.folders, self.files, "input", key, region)
            processed_region_samples = load_dataframe(self.context.working, region_key)
            region_samples = self.transform_sample_voxels(
                processed_region_samples, margins, processed_samples
            )
            region_samples["region"] = region

            samples = samples.merge(region_samples, on=["id", "x", "y", "z"], how="left")
            samples["region"].fillna("DEFAULT", inplace=True)

        # Filter samples for valid samples.
        samples = self.filter_valid_samples(samples)

        # Initialize outputs.
        cells: List[Dict] = []
        locations: List[Dict] = []
        samples_by_id = samples.groupby("id")

        # Iterate through each cell in samples and convert.
        for i, (cell_id, group) in enumerate(samples_by_id):
            cell_reference = self.filter_cell_reference(cell_id, reference)
            cells.append(self.convert_to_cell(i + 1, group, cell_reference))
            locations.append(self.convert_to_location(i + 1, group))

        # Save converted ARCADE files.
        cells_key = make_full_key(self.folders, self.files, "cells", key)
        save_json(self.context.working, cells_key, cells)

        locations_key = make_full_key(self.folders, self.files, "locations", key)
        save_json(self.context.working, locations_key, locations)

        # Create and save setup file.
        init = len(samples["id"].unique())
        bounds = self.calculate_sample_bounds(processed_samples, margins)
        regions = samples["regions"].unique() if "regions" in samples else None
        setup = self.make_setup_file(init, bounds, regions)
        setup_key = make_full_key(self.folders, self.files, "setup", key)
        save_buffer(self.context.working, setup_key, io.BytesIO(setup))

    @staticmethod
    def transform_sample_voxels(
        samples: pd.DataFrame,
        margins: Tuple[int, int, int],
        reference: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Transforms samples into centered voxel coordinates.

        Parameters
        ----------
        samples
            Sample cell ids and coordinates.
        margins
            Margin size in x, y, and z directions.
        reference
            Reference samples used to calculate transformation.

        Returns
        -------
        :
            Sample cell ids and coordinates.
        """
        if reference is None:
            reference = samples

        steps = ProcessSamples.get_step_sizes(reference)
        minimums = ProcessSamples.get_sample_minimums(reference)
        offsets = np.add(-1 * np.divide(minimums, steps), margins) + 1

        coordinates = samples[["x", "y", "z"]].values / steps + offsets
        coordinates = coordinates.astype("int64")

        transformed = pd.DataFrame(coordinates, columns=["x", "y", "z"])
        transformed.insert(0, "id", samples["id"])

        return transformed

    @staticmethod
    def calculate_sample_bounds(
        samples: pd.DataFrame, margins: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        """
        Calculate transformed sample bounds including margin.

        Parameters
        ----------
        samples
            Sample cell ids and coordinates.
        margins
            Margin size in x, y, and z directions.

        Returns
        -------
        :
            Bounds in x, y, and z directions.
        """
        steps = ProcessSamples.get_step_sizes(samples)
        mins = ProcessSamples.get_sample_minimums(samples)
        maxs = ProcessSamples.get_sample_maximums(samples)

        bound_x, bound_y, bound_z = (np.subtract(maxs, mins) / steps) + np.multiply(2, margins) + 3
        bounds = (bound_x, bound_y, bound_z)
        return bounds

    @staticmethod
    def make_setup_file(
        init: int,
        bounds: Tuple[int, int, int],
        regions: Optional[List[str]] = None,
        terms: Tuple = POTTS_TERMS,
    ) -> bytes:
        """
        Create ARCADE setup file for converted samples.

        Parameters
        ----------
        init
            Number of initial cells.
        bounds
            Bounds in x, y, and z directions.
        regions
            List of regions.
        terms
            List of Potts Hamiltonian terms for setup file, default = ``POTTS_TERMS``.

        Returns
        -------
            Contents of ARCADE setup file.
        """
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
                "ds": "1",
                "ticks": "1",
                "length": str(int(bounds[0])),
                "width": str(int(bounds[1])),
                "height": str(int(bounds[2])),
            },
        )

        potts = ET.SubElement(series, "potts")
        for term in terms:
            ET.SubElement(potts, "potts.term", {"id": term})

        agents = ET.SubElement(series, "agents")
        populations = ET.SubElement(agents, "populations")
        population = ET.SubElement(populations, "population", {"id": "X", "init": str(init)})

        if regions is not None:
            for region in regions:
                ET.SubElement(population, "population.region", {"id": region})

        ET.indent(root, space="    ", level=0)
        return ET.tostring(root)

    @staticmethod
    def filter_valid_samples(samples: pd.DataFrame) -> pd.DataFrame:
        """
        Filters samples for valid cell ids.

        Filter conditions include:

        - Each cell must have at least one voxel assigned to each specified region

        Parameters
        ----------
        samples
            Sample cell ids and coordinates.

        Returns
        -------
        :
            Valid Sample cell ids and coordinates.
        """
        if "region" in samples.columns:
            # Ensure that each cell has all regions.
            num_regions = len(samples.region.unique())
            samples = samples.groupby("id").filter(lambda x: len(x.region.unique()) == num_regions)

        return samples.reset_index(drop=True)

    @staticmethod
    def filter_cell_reference(cell_id: int, reference: pd.DataFrame) -> dict:
        """
        Filters reference data for given cell id.

        Parameters
        ----------
        cell_id
            Unique cell id.
        reference
            Reference data for conversion.

        Returns
        -------
        :
            Reference data for given cell id.
        """
        cell_reference = reference[reference.id == cell_id].squeeze()
        cell_reference = cell_reference.to_dict() if not cell_reference.empty else {}
        return cell_reference

    @staticmethod
    def convert_to_cell(cell_id: int, samples: pd.DataFrame, reference: dict) -> dict:
        """
        Convert samples to ARCADE .CELLS json format.

        Parameters
        ----------
        cell_id
            Unique cell id.
        samples
            Sample cell ids and coordinates.
        reference
            Reference data for conversion.

        Returns
        -------
        :
            Dictionary in ARCADE .CELLS json format.
        """
        volume = len(samples)

        critical_volume = (
            reference["volume"]
            if "volume" in reference
            else ConvertARCADE.get_cell_critical_volume(samples)
        )
        critical_height = (
            reference["height"]
            if "height" in reference
            else ConvertARCADE.get_cell_critical_height(samples)
        )

        state = ConvertARCADE.get_cell_state(volume, critical_volume)

        cell = {
            "id": cell_id,
            "parent": 0,
            "pop": 1,
            "age": 0,
            "divisions": 0,
            "state": state.split("_")[0],
            "phase": state,
            "voxels": volume,
            "criticals": [critical_volume, critical_height],
        }

        if "region" in samples.columns:
            regions: List[dict] = []

            for region, region_samples in samples.groupby("region"):
                region_critical_volume = (
                    reference[f"volume.{region}"]
                    if f"volume.{region}" in reference
                    else ConvertARCADE.get_cell_critical_volume(region_samples, str(region))
                )
                region_critical_height = (
                    reference[f"height.{region}"]
                    if f"height.{region}" in reference
                    else ConvertARCADE.get_cell_critical_height(region_samples, str(region))
                )

                regions.append(
                    {
                        "region": region,
                        "voxels": len(region_samples),
                        "criticals": [region_critical_volume, region_critical_height],
                    }
                )

            cell.update({"regions": regions})

        return cell

    @staticmethod
    def convert_to_location(cell_id: int, samples: pd.DataFrame) -> dict:
        """
        Convert samples to ARCADE .LOCATIONS json format.

        Parameters
        ----------
        cell_id
            Unique cell id.
        samples
            Sample cell ids and coordinates.

        Returns
        -------
        :
            Dictionary in ARCADE .LOCATIONS json format.
        """
        center = ConvertARCADE.get_location_center(samples)

        if "region" in samples.columns:
            voxels = [
                {"region": region, "voxels": ConvertARCADE.get_location_voxels(samples, region)}
                for region in samples["region"].unique()
            ]
        else:
            voxels = [{"region": "UNDEFINED", "voxels": ConvertARCADE.get_location_voxels(samples)}]

        location = {
            "id": cell_id,
            "center": center,
            "location": voxels,
        }

        return location

    @staticmethod
    def get_cell_critical_volume(
        samples: pd.DataFrame,
        region: str = "DEFAULT",
        avgs: Optional[Dict[str, float]] = None,
        stds: Optional[Dict[str, float]] = None,
        critical_avgs: Optional[Dict[str, float]] = None,
        critical_stds: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Estimates critical cell volume based on samples.

        Parameters
        ----------
        samples
            Sample cell ids and coordinates.
        region
            Region key.
        avgs
            Volume distribution averages, default = ``VOLUME_AVGS``.
        stds
            Volume distribution standard deviations, default = ``VOLUME_STDS``.
        critical_avgs
            Critical volume distribution averages, default = ``CRITICAL_VOLUME_AVGS``.
        critical_stds
            Critical volume distribution deviations, default = ``CRITICAL_VOLUME_STDS``.

        Returns
        -------
        :
            Estimated critical cell volume.
        """
        if avgs is None:
            avgs = VOLUME_AVGS

        if stds is None:
            stds = VOLUME_STDS

        if critical_avgs is None:
            critical_avgs = CRITICAL_VOLUME_AVGS

        if critical_stds is None:
            critical_stds = CRITICAL_VOLUME_STDS

        volume = len(samples)
        z_scored_volume = (volume - avgs[region]) / stds[region]
        critical_volume = z_scored_volume * critical_stds[region] + critical_avgs[region]
        return critical_volume

    @staticmethod
    def get_cell_critical_height(
        samples: pd.DataFrame,
        region: str = "DEFAULT",
        avgs: Optional[Dict[str, float]] = None,
        stds: Optional[Dict[str, float]] = None,
        critical_avgs: Optional[Dict[str, float]] = None,
        critical_stds: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Estimates critical cell height based on samples.

        Parameters
        ----------
        samples
            Sample cell ids and coordinates.
        region
            Region key.
        avgs
            Height distribution averages, default = ``HEIGHT_AVGS``.
        stds
            Height distribution standard deviations, default = ``HEIGHT_STDS``.
        critical_avgs
            Critical height distribution averages, default = ``CRITICAL_HEIGHT_AVGS``.
        critical_stds
            Critical height distribution deviations, default = ``CRITICAL_HEIGHT_STDS``.

        Returns
        -------
        :
            Estimated critical cell volume.
        """
        if avgs is None:
            avgs = HEIGHT_AVGS

        if stds is None:
            stds = HEIGHT_STDS

        if critical_avgs is None:
            critical_avgs = CRITICAL_HEIGHT_AVGS

        if critical_stds is None:
            critical_stds = CRITICAL_HEIGHT_STDS

        height = samples.z.max() - samples.z.min()
        z_scored_height = (height - avgs[region]) / stds[region]
        critical_height = z_scored_height * critical_stds[region] + critical_avgs[region]
        return critical_height

    @staticmethod
    def get_cell_state(
        volume: float,
        critical_volume: float,
        threshold_fractions: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Estimates cell state based on cell volume.

        The threshold fractions dictionary defines the monotonic thresholds
        between different cell states.
        For a given volume v, critical volume V, and states X1, X2, ..., XN with
        corresponding, monotonic threshold fractions f1, f2, ..., fN, a cell is
        assigned state Xi such that [f(i - 1) * V] <= v < [fi * V].
        Cells with v < f1 * V are assigned state X1.
        Cells with v > fN * V are assigned state XN.

        Parameters
        ----------
        volume
            Current cell volume.
        critical_volume
            Critical cell volume.
        threshold_fractions
            Critical volume fractions defining threshold between states,
            default = ``CELL_STATE_THRESHOLD_FRACTIONS``.

        Returns
        -------
        :
            Cell state.
        """
        if threshold_fractions is None:
            threshold_fractions = CELL_STATE_THRESHOLD_FRACTIONS

        thresholds = [fraction * critical_volume for fraction in threshold_fractions.values()]
        states = list(threshold_fractions.keys())

        index = next((ind for ind, thresh in enumerate(thresholds) if thresh > volume), -1)
        return states[index]

    @staticmethod
    def get_location_center(samples: pd.DataFrame) -> Tuple[int, int, int]:
        """
        Gets coordinates of center of samples.

        Parameters
        ----------
        samples
            Sample cell ids and coordinates.

        Returns
        -------
        :
            Center voxel.
        """
        center_x = int(samples["x"].mean())
        center_y = int(samples["y"].mean())
        center_z = int(samples["z"].mean())
        center = (center_x, center_y, center_z)
        return center

    @staticmethod
    def get_location_voxels(
        samples: pd.DataFrame, region: Optional[str] = None
    ) -> List[Tuple[int, int, int]]:
        """
        Get list of voxel coordinates from samples dataframe.

        Parameters
        ----------
        samples
            Sample cell ids and coordinates.
        region
            Region key.

        Returns
        -------
        :
            List of voxel coordinates.
        """
        if region is not None:
            region_samples = samples[samples["region"] == region]
            voxels_x = region_samples["x"]
            voxels_y = region_samples["y"]
            voxels_z = region_samples["z"]
        else:
            voxels_x = samples["x"]
            voxels_y = samples["y"]
            voxels_z = samples["z"]

        return list(zip(voxels_x, voxels_y, voxels_z))
