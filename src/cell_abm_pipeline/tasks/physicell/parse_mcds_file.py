import re
import tarfile
import tempfile
from typing import Union

import numpy as np
import pandas as pd
from simulariumio.physicell.dep.pyMCDS import pyMCDS

COLUMN_NAMES = [
    "ID",
    "TICK",
    "NUM_SUBCELLS",
    "TOTAL_VOLUME",
    "CENTER_X",
    "MIN_X",
    "MAX_X",
    "CENTER_Y",
    "MIN_Y",
    "MAX_Y",
    "CENTER_Z",
    "MIN_Z",
    "MAX_Z",
]


def parse_mcds_file(tar: tarfile.TarFile, max_owner_cells: int = 10000) -> pd.DataFrame:
    file_mapping: dict[str, dict] = {}

    for member in tar.getmembers():
        match = re.match(r"output([0-9]+)[_]*([A-z0-9]*\.[a-z]+)", member.name)

        if match is None:
            continue

        timepoint, extension = match.groups()

        if timepoint not in file_mapping:
            file_mapping[timepoint] = {}

        file_mapping[timepoint][extension] = member

    all_cells: list[list[Union[str, int, float]]] = []

    for timepoint, files in file_mapping.items():
        with tempfile.TemporaryDirectory() as temp_directory:
            tar.extract("initial_mesh0.mat", path=temp_directory)

            for file in files.values():
                tar.extract(file, path=temp_directory)

            mcds = pyMCDS(files[".xml"].name, False, temp_directory)
            subcell_df = mcds.get_cell_df()

        cells = parse_subcell_timepoint(int(timepoint), subcell_df, max_owner_cells)
        all_cells = all_cells + cells

    cells_df = pd.DataFrame(all_cells, columns=COLUMN_NAMES)

    return cells_df


def calculate_radius_from_volume(total_volume: float) -> float:
    return np.cbrt(3.0 / 4.0 * total_volume / np.pi)


def parse_subcell_timepoint(timepoint: int, subcell_df: pd.DataFrame, max_owner_cells: int) -> list:
    all_cells = []

    for cell_id, subcells in subcell_df.groupby("cell_type"):
        owner_cell_id = int(cell_id)

        while owner_cell_id - max_owner_cells > 0:
            owner_cell_id -= max_owner_cells

        if owner_cell_id == 0:
            continue

        total_volume = subcells["total_volume"].sum()
        positions = parse_subcell_positions(subcells)
        cell = [owner_cell_id, timepoint, len(subcells), total_volume] + positions

        all_cells.append(cell)

    return all_cells


def parse_subcell_positions(subcells: pd.DataFrame) -> list:
    parsed = []

    for coordinate in ["x", "y", "z"]:
        parsed.append(subcells[f"position_{coordinate}"].mean())

        min_subcell = subcells.loc[subcells[f"position_{coordinate}"].idxmin()]
        min_radius = calculate_radius_from_volume(min_subcell["total_volume"])
        parsed.append(min_subcell[f"position_{coordinate}"] - min_radius)

        max_subcell = subcells.loc[subcells[f"position_{coordinate}"].idxmax()]
        max_radius = calculate_radius_from_volume(max_subcell["total_volume"])
        parsed.append(max_subcell[f"position_{coordinate}"] + max_radius)

    return parsed
