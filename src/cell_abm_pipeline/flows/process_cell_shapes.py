import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from io_collection.keys import check_key, make_key, remove_key
from io_collection.load import load_dataframe, load_tar
from io_collection.save import save_dataframe, save_tar
from prefect import flow


@dataclass
class ParametersConfig:
    frames: list[int]

    region: Optional[str] = None


@dataclass
class ContextConfig:
    working_location: str


@dataclass
class SeriesConfig:
    name: str

    seeds: list[int]

    conditions: list[dict]


@flow(name="process-cell-shapes")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    # Iterate through conditions and seeds to merge contents of individual
    # frames into a single csv. If merged csv exists and the specified frame
    # does not exist in the csv, the frame is appended. If the merged csv exists
    # and specified frame exists in the csv, the frame is skipped.
    run_flow_merge_shapes(context, series, parameters)

    # Iterate through conditions and seeds to combine and compress individual
    # frames into a .tar.xz archive. If the archive exists and the specified
    # frame is not in the archive, the frame is appended. If the archive exists
    # and specified frame exists in the archive, the frame is skipped.
    run_flow_compress_shapes(context, series, parameters)

    # Iterate through conditions and seeds to remove individual frames if the
    # frame exists in the corresponding .tar.xz archive.
    run_flow_remove_shapes(context, series, parameters)


@flow(name="process-cell-shapes_merge-shapes")
def run_flow_merge_shapes(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    analysis_key = make_key(series.name, "analysis", "analysis.COEFFS")
    region_key = f"_{parameters.region}" if parameters.region is not None else ""

    for condition in series.conditions:
        for seed in series.seeds:
            series_key = f"{series.name}_{condition['key']}_{seed:04d}"
            coeff_key = make_key(analysis_key, f"{series_key}{region_key}.COEFFS.csv")
            coeff_key_exists = check_key(context.working_location, coeff_key)

            existing_frames = []
            if coeff_key_exists:
                existing_coeffs = load_dataframe(context.working_location, coeff_key)
                existing_frames = list(existing_coeffs["TICK"].unique())

            frame_coeffs = []

            for frame in parameters.frames:
                if frame in existing_frames:
                    continue

                frame_key = make_key(
                    analysis_key, f"{series_key}_{frame:06d}{region_key}.COEFFS.csv"
                )
                frame_coeffs.append(load_dataframe(context.working_location, frame_key))

            if not frame_coeffs:
                return

            coeff_dataframe = pd.concat(frame_coeffs, ignore_index=True)

            if coeff_key_exists:
                coeff_dataframe = pd.concat([existing_coeffs, coeff_dataframe], ignore_index=True)

            save_dataframe(context.working_location, coeff_key, coeff_dataframe, index=False)


@flow(name="process-cell-shapes_compress-shapes")
def run_flow_compress_shapes(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    analysis_key = make_key(series.name, "analysis", "analysis.COEFFS")
    region_key = f"_{parameters.region}" if parameters.region is not None else ""

    for condition in series.conditions:
        for seed in series.seeds:
            series_key = f"{series.name}_{condition['key']}_{seed:04d}"
            coeff_key = make_key(analysis_key, f"{series_key}{region_key}.COEFFS.tar.xz")
            coeff_key_exists = check_key(context.working_location, coeff_key)

            existing_frames = []
            if coeff_key_exists:
                existing_coeffs = load_tar(context.working_location, coeff_key)
                existing_frames = [
                    int(re.findall(r"[0-9]{6}", member.name)[0])
                    for member in existing_coeffs.getmembers()
                ]

            contents = []

            for frame in parameters.frames:
                if frame in existing_frames:
                    continue

                frame_key = make_key(
                    analysis_key, f"{series_key}_{frame:06d}{region_key}.COEFFS.csv"
                )

                contents.append(frame_key)

            if not contents:
                return

            save_tar(context.working_location, coeff_key, contents)


@flow(name="process-cell-shapes_remove-shapes")
def run_flow_remove_shapes(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    analysis_key = make_key(series.name, "analysis", "analysis.COEFFS")
    region_key = f"_{parameters.region}" if parameters.region is not None else ""

    for condition in series.conditions:
        for seed in series.seeds:
            series_key = f"{series.name}_{condition['key']}_{seed:04d}"
            coeff_key = make_key(analysis_key, f"{series_key}{region_key}.COEFFS.tar.xz")
            coeff_key_exists = check_key(context.working_location, coeff_key)

            if not coeff_key_exists:
                return

            existing_coeffs = load_tar(context.working_location, coeff_key)

            for member in existing_coeffs.getmembers():
                frame_key = make_key(analysis_key, member.name)

                if check_key(context.working_location, frame_key):
                    remove_key(context.working_location, frame_key)
