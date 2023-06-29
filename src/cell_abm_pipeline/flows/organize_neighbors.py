"""
Workflow for organizing neighbor connections analysis files.
"""

import re
from dataclasses import dataclass

import pandas as pd
from io_collection.keys import check_key, make_key, remove_key
from io_collection.load import load_dataframe, load_tar
from io_collection.save import save_dataframe, save_tar
from prefect import flow


@dataclass
class ParametersConfig:
    """Parameter configuration for organize neighbors flow."""

    frames: list[int]


@dataclass
class ContextConfig:
    """Context configuration for organize neighbors flow."""

    working_location: str


@dataclass
class SeriesConfig:
    """Series configuration for organize neighbors flow."""

    name: str

    seeds: list[int]

    conditions: list[dict]


@flow(name="organize-neighbors")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    # Iterate through conditions and seeds to merge contents of individual
    # frames into a single csv. If merged csv exists and the specified frame
    # does not exist in the csv, the frame is appended. If the merged csv exists
    # and specified frame exists in the csv, the frame is skipped.
    run_flow_merge_neighbors(context, series, parameters)

    # Iterate through conditions and seeds to combine and compress individual
    # frames into a .tar.xz archive. If the archive exists and the specified
    # frame is not in the archive, the frame is appended. If the archive exists
    # and specified frame exists in the archive, the frame is skipped.
    run_flow_compress_neighbors(context, series, parameters)

    # Iterate through conditions and seeds to remove individual frames if the
    # frame exists in the corresponding .tar.xz archive.
    run_flow_remove_neighbors(context, series, parameters)


@flow(name="organize-neighbors_merge-neighbors")
def run_flow_merge_neighbors(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    analysis_key = make_key(series.name, "analysis", "analysis.NEIGHBORS")

    for condition in series.conditions:
        for seed in series.seeds:
            series_key = f"{series.name}_{condition['key']}_{seed:04d}"
            neighbor_key = make_key(analysis_key, f"{series_key}.NEIGHBORS.csv")
            neighbor_key_exists = check_key(context.working_location, neighbor_key)

            existing_frames = []
            if neighbor_key_exists:
                existing_neighbors = load_dataframe(context.working_location, neighbor_key)
                existing_frames = list(existing_neighbors["TICK"].unique())

            frame_neighbors = []

            for frame in parameters.frames:
                if frame in existing_frames:
                    continue

                frame_key = make_key(analysis_key, f"{series_key}_{frame:06d}.NEIGHBORS.csv")

                frame_neighbors.append(load_dataframe(context.working_location, frame_key))

            if not frame_neighbors:
                continue

            neighbor_dataframe = pd.concat(frame_neighbors, ignore_index=True)

            if neighbor_key_exists:
                neighbor_dataframe = pd.concat(
                    [existing_neighbors, neighbor_dataframe], ignore_index=True
                )

            save_dataframe(context.working_location, neighbor_key, neighbor_dataframe, index=False)


@flow(name="organize-neighbors_compress-neighbors")
def run_flow_compress_neighbors(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    analysis_key = make_key(series.name, "analysis", "analysis.NEIGHBORS")

    for condition in series.conditions:
        for seed in series.seeds:
            series_key = f"{series.name}_{condition['key']}_{seed:04d}"
            neighbor_key = make_key(analysis_key, f"{series_key}.NEIGHBORS.tar.xz")
            neighbor_key_exists = check_key(context.working_location, neighbor_key)

            existing_frames = []
            if neighbor_key_exists:
                existing_neighbors = load_tar(context.working_location, neighbor_key)
                existing_frames = [
                    int(re.findall(r"[0-9]{6}", member.name)[0])
                    for member in existing_neighbors.getmembers()
                ]

            contents = []

            for frame in parameters.frames:
                if frame in existing_frames:
                    continue

                frame_key = make_key(analysis_key, f"{series_key}_{frame:06d}.NEIGHBORS.csv")

                contents.append(frame_key)

            if not contents:
                continue

            save_tar(context.working_location, neighbor_key, contents)


@flow(name="organize-neighbors_remove-neighbors")
def run_flow_remove_neighbors(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    analysis_key = make_key(series.name, "analysis", "analysis.NEIGHBORS")

    for condition in series.conditions:
        for seed in series.seeds:
            series_key = f"{series.name}_{condition['key']}_{seed:04d}"
            neighbor_key = make_key(analysis_key, f"{series_key}.NEIGHBORS.tar.xz")
            neighbor_key_exists = check_key(context.working_location, neighbor_key)

            if not neighbor_key_exists:
                continue

            existing_neighbors = load_tar(context.working_location, neighbor_key)

            for member in existing_neighbors.getmembers():
                frame_key = make_key(analysis_key, member.name)

                if check_key(context.working_location, frame_key):
                    remove_key(context.working_location, frame_key)
