from dataclasses import dataclass
from typing import Optional

from container_collection.fargate import (
    make_fargate_task,
    register_fargate_task,
    submit_fargate_task,
)
from io_collection.keys import check_key, make_key
from io_collection.load import load_dataframe
from prefect import flow


@dataclass
class ParametersConfig:
    image: str

    frames: list[int]

    scale: int

    region: Optional[str] = None

    chunk_size: int = 100


@dataclass
class ContextConfig:
    working_location: str

    account: str

    region: str

    user: str

    vcpus: int

    memory: int

    cluster: str

    security_groups: str

    subnets: str


@dataclass
class SeriesConfig:
    name: str

    seeds: list[int]

    conditions: list[dict]


@flow(name="analyze-cell-shapes")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    task_definition = make_fargate_task(
        "cell_shape",
        parameters.image,
        context.account,
        context.region,
        context.user,
        context.vcpus,
        context.memory,
    )
    task_definition_arn = register_fargate_task(task_definition)

    for condition in series.conditions:
        for seed in series.seeds:
            series_key = f"{series.name}_{condition['key']}_{seed:04d}"
            coeff_key = make_key(
                series.name, "analysis", "analysis.COEFFS", f"{series_key}.COEFFS.csv"
            )
            coeff_key_exists = check_key(context.working_location, coeff_key)

            existing_frames = []
            if coeff_key_exists:
                existing_coeffs = load_dataframe(
                    context.working_location, coeff_key, usecols=["TICK"]
                )
                existing_frames = list(existing_coeffs["TICK"].unique())

            results_key = make_key(series.name, "results", f"{series_key}.csv")
            results = load_dataframe(context.working_location, results_key)

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
                frame_key_exists = check_key(context.working_location, frame_key)

                if frame_key_exists:
                    continue

                total = results[results["TICK"] == frame].shape[0]
                offsets = list(range(0, total, parameters.chunk_size))

                for offset in offsets:
                    calculate_coefficients_command = [
                        "abmpipe",
                        "calculate-coefficients",
                        "::",
                        f"parameters.key={condition['key']}",
                        f"parameters.seed={seed}",
                        f"parameters.scale={parameters.scale}",
                        f"parameters.frame={frame}",
                        f"context.working_location={context.working_location}",
                        f"series.name={series.name}",
                        f"parameters.offset={offset}",
                        f"parameters.chunk={parameters.chunk_size}",
                    ]

                    if parameters.region:
                        calculate_coefficients_command.append(
                            f"parameters.region={parameters.region}"
                        )

                    submit_fargate_task(
                        "cell_shape",
                        task_definition_arn,
                        context.user,
                        context.cluster,
                        context.security_groups.split(":"),
                        context.subnets.split(":"),
                        calculate_coefficients_command,
                    )
