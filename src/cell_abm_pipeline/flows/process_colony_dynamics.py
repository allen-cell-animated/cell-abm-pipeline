import re
from dataclasses import dataclass

import pandas as pd
from io_collection.keys import check_key, make_key, remove_key
from io_collection.load import load_dataframe, load_tar
from io_collection.save import save_dataframe, save_tar
from prefect import flow


@dataclass
class ParametersConfig:
    frames: list[int]


@dataclass
class ContextConfig:
    working_location: str


@dataclass
class SeriesConfig:
    name: str

    seeds: list[int]

    conditions: list[dict]


@flow(name="process-colony-dynamics")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    run_flow_merge_colonies(context, series, parameters)

    run_flow_compress_colonies(context, series, parameters)

    run_flow_remove_colonies(context, series, parameters)


@flow(name="process-colony-dynamics_merge-colonies")
def run_flow_merge_colonies(
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
                return

            neighbor_dataframe = pd.concat(frame_neighbors, ignore_index=True)

            if neighbor_key_exists:
                neighbor_dataframe = pd.concat(
                    [existing_neighbors, neighbor_dataframe], ignore_index=True
                )

            save_dataframe(context.working_location, neighbor_key, neighbor_dataframe, index=False)


@flow(name="process-colony-dynamics_compress-colonies")
def run_flow_compress_colonies(
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
                return

            save_tar(context.working_location, neighbor_key, contents)


@flow(name="process-colony-dynamics_remove-colonies")
def run_flow_remove_colonies(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    analysis_key = make_key(series.name, "analysis", "analysis.NEIGHBORS")

    for condition in series.conditions:
        for seed in series.seeds:
            series_key = f"{series.name}_{condition['key']}_{seed:04d}"
            neighbor_key = make_key(analysis_key, f"{series_key}.NEIGHBORS.tar.xz")
            neighbor_key_exists = check_key(context.working_location, neighbor_key)

            if not neighbor_key_exists:
                return

            existing_neighbors = load_tar(context.working_location, neighbor_key)

            for member in existing_neighbors.getmembers():
                frame_key = make_key(analysis_key, member.name)

                if check_key(context.working_location, frame_key):
                    remove_key(context.working_location, frame_key)
