import re
from dataclasses import dataclass
from typing import Optional
from prefect import flow
import pandas as pd

from io_collection.keys import make_key, check_key, remove_key
from io_collection.load import load_dataframe, load_tar
from io_collection.save import save_dataframe, save_tar


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


@flow(name="merge-cell-shape")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    merge_cell_shapes(context, series, parameters)
    compress_cell_shapes(context, series, parameters)
    remove_cell_shapes(context, series, parameters)


@flow(name="merge-cell-shapes")
def merge_cell_shapes(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    for condition in series.conditions:
        for seed in series.seeds:
            series_key = f"{series.name}_{condition['key']}_{seed:04d}"
            coeff_key = make_key(
                series.name, "analysis", "analysis.COEFFS", f"{series_key}.COEFFS.csv"
            )
            coeff_key_exists = check_key(context.working_location, coeff_key)

            existing_frames = []
            if coeff_key_exists:
                existing_coeffs = load_dataframe(context.working_location, coeff_key)
                existing_frames = list(existing_coeffs["TICK"].unique())

            frame_coeffs = []

            for frame in parameters.frames:
                if frame in existing_frames:
                    continue

                region_key = f"_{parameters.region}" if parameters.region is not None else ""
                frame_key = make_key(
                    series.name,
                    "analysis",
                    "analysis.COEFFS",
                    f"{series_key}_{frame:06d}{region_key}.COEFFS.csv",
                )

                frame_coeffs.append(load_dataframe(context.working_location, frame_key))

            if not frame_coeffs:
                continue

            coeff_dataframe = pd.concat(frame_coeffs, ignore_index=True)

            if coeff_key_exists:
                coeff_dataframe = pd.concat([existing_coeffs, coeff_dataframe], ignore_index=True)

            coeff_key_timestamp = make_key(
                series.name, "analysis", "analysis.COEFFS", f"{series_key}.COEFFS.csv"
            )
            save_dataframe(
                context.working_location, coeff_key_timestamp, coeff_dataframe, index=False
            )


@flow(name="compress-cell-shapes")
def compress_cell_shapes(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    for condition in series.conditions:
        for seed in series.seeds:
            series_key = f"{series.name}_{condition['key']}_{seed:04d}"
            coeff_key = make_key(
                series.name, "analysis", "analysis.COEFFS", f"{series_key}.COEFFS.tar.xz"
            )
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

                region_key = f"_{parameters.region}" if parameters.region is not None else ""
                frame_key = make_key(
                    series.name,
                    "analysis",
                    "analysis.COEFFS",
                    f"{series_key}_{frame:06d}{region_key}.COEFFS.csv",
                )

                contents.append(frame_key)

            if not contents:
                continue

            save_tar(context.working_location, coeff_key, contents)


@flow(name="remove-cell-shapes")
def remove_cell_shapes(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    for condition in series.conditions:
        for seed in series.seeds:
            series_key = f"{series.name}_{condition['key']}_{seed:04d}"
            coeff_key = make_key(
                series.name, "analysis", "analysis.COEFFS", f"{series_key}.COEFFS.tar.xz"
            )
            coeff_key_exists = check_key(context.working_location, coeff_key)

            if not coeff_key_exists:
                continue

            existing_coeffs = load_tar(context.working_location, coeff_key)

            for member in existing_coeffs.getmembers():
                frame_key = make_key(series.name, "analysis", "analysis.COEFFS", member.name)

                if check_key.fn(context.working_location, frame_key):
                    remove_key(context.working_location, frame_key)
