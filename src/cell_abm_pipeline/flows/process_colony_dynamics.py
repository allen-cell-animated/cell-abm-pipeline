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
    for condition in series.conditions:
        for seed in series.seeds:
            merge_colony_dynamics(
                series.name,
                condition["key"],
                seed,
                context.working_location,
                parameters.frames,
            )
            compress_colony_dynamics(
                series.name,
                condition["key"],
                seed,
                context.working_location,
                parameters.frames,
            )
            remove_colony_dynamics(
                series.name,
                condition["key"],
                seed,
                context.working_location,
            )


@flow(name="merge-colony-dynamics")
def merge_colony_dynamics(name, condition, seed, working_location, frames) -> None:
    series_key = f"{name}_{condition}_{seed:04d}"
    coeff_key = make_key(name, "analysis", "analysis.NEIGHBORS", f"{series_key}.NEIGHBORS.csv")
    coeff_key_exists = check_key(working_location, coeff_key)

    existing_frames = []
    if coeff_key_exists:
        existing_neighbors = load_dataframe(working_location, coeff_key)
        existing_frames = list(existing_neighbors["TICK"].unique())

    frame_neighbors = []

    for frame in frames:
        if frame in existing_frames:
            continue

        frame_key = make_key(
            name, "analysis", "analysis.NEIGHBORS", f"{series_key}_{frame:06d}.NEIGHBORS.csv"
        )

        frame_neighbors.append(load_dataframe(working_location, frame_key))

    if not frame_neighbors:
        return

    coeff_dataframe = pd.concat(frame_neighbors, ignore_index=True)

    if coeff_key_exists:
        coeff_dataframe = pd.concat([existing_neighbors, coeff_dataframe], ignore_index=True)

    save_dataframe(working_location, coeff_key, coeff_dataframe, index=False)


@flow(name="compress-colony-dynamics")
def compress_colony_dynamics(name, condition, seed, working_location, frames) -> None:
    series_key = f"{name}_{condition}_{seed:04d}"
    coeff_key = make_key(name, "analysis", "analysis.NEIGHBORS", f"{series_key}.NEIGHBORS.tar.xz")
    coeff_key_exists = check_key(working_location, coeff_key)

    existing_frames = []
    if coeff_key_exists:
        existing_neighbors = load_tar(working_location, coeff_key)
        existing_frames = [
            int(re.findall(r"[0-9]{6}", member.name)[0])
            for member in existing_neighbors.getmembers()
        ]

    contents = []

    for frame in frames:
        if frame in existing_frames:
            continue

        frame_key = make_key(
            name, "analysis", "analysis.NEIGHBORS", f"{series_key}_{frame:06d}.NEIGHBORS.csv"
        )

        contents.append(frame_key)

    if not contents:
        return

    save_tar(working_location, coeff_key, contents)


@flow(name="remove-colony-dynamics")
def remove_colony_dynamics(name, condition, seed, working_location) -> None:
    series_key = f"{name}_{condition}_{seed:04d}"
    coeff_key = make_key(name, "analysis", "analysis.NEIGHBORS", f"{series_key}.NEIGHBORS.tar.xz")
    coeff_key_exists = check_key(working_location, coeff_key)

    if not coeff_key_exists:
        return

    existing_neighbors = load_tar(working_location, coeff_key)

    for member in existing_neighbors.getmembers():
        frame_key = make_key(name, "analysis", "analysis.NEIGHBORS", member.name)

        if check_key(working_location, frame_key):
            remove_key(working_location, frame_key)
