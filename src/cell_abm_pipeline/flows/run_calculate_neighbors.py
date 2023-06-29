"""
Workflow for running calculate neighbor connections flow across conditions and seeds.
"""

import copy
from dataclasses import dataclass

from container_collection.fargate import (
    make_fargate_task,
    register_fargate_task,
    submit_fargate_task,
)
from io_collection.keys import check_key, make_key
from io_collection.load import load_dataframe
from prefect import flow

from cell_abm_pipeline.__config__ import make_dotlist_from_config
from cell_abm_pipeline.flows.calculate_neighbors import (
    ContextConfig as CalculateNeighborsContextConfig,
)
from cell_abm_pipeline.flows.calculate_neighbors import (
    ParametersConfig as CalculateNeighborsParametersConfig,
)
from cell_abm_pipeline.flows.calculate_neighbors import (
    SeriesConfig as CalculateNeighborsSeriesConfig,
)

CALCULATE_NEIGHBORS_COMMAND = ["abmpipe", "calculate-neighbors", "::"]


@dataclass
class ParametersConfig:
    image: str

    frames: list[int]

    calculate_neighbors: CalculateNeighborsParametersConfig

    submit_tasks: bool = True


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


@flow(name="run-calculate-neighbors")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    analysis_key = make_key(series.name, "analysis", "analysis.NEIGHBORS")

    task_definition = make_fargate_task(
        "calculate_neighbors",
        parameters.image,
        context.account,
        context.region,
        context.user,
        context.vcpus,
        context.memory,
    )
    task_definition_arn = register_fargate_task(task_definition)

    context_config = CalculateNeighborsContextConfig(working_location=context.working_location)
    series_config = CalculateNeighborsSeriesConfig(name=series.name)

    for condition in series.conditions:
        for seed in series.seeds:
            series_key = f"{series.name}_{condition['key']}_{seed:04d}"
            neighbor_key = make_key(analysis_key, f"{series_key}.NEIGHBORS.csv")
            neighbor_key_exists = check_key(context.working_location, neighbor_key)

            existing_frames = []
            if neighbor_key_exists:
                existing_neighbors = load_dataframe(
                    context.working_location, neighbor_key, usecols=["TICK"]
                )
                existing_frames = list(existing_neighbors["TICK"].unique())

            for frame in parameters.frames:
                if frame in existing_frames:
                    continue

                frame_key = make_key(analysis_key, f"{series_key}_{frame:06d}.NEIGHBORS.csv")
                frame_key_exists = check_key(context.working_location, frame_key)

                if frame_key_exists:
                    continue

                parameters_config = copy.deepcopy(parameters.calculate_neighbors)
                parameters_config.key = parameters_config.key % condition["key"]
                parameters_config.seed = seed
                parameters_config.frame = frame

                config = {
                    "context": context_config,
                    "series": series_config,
                    "parameters": parameters_config,
                }

                calculate_neighbors_command = (
                    CALCULATE_NEIGHBORS_COMMAND + make_dotlist_from_config(config)
                )

                if parameters.submit_tasks:
                    submit_fargate_task.with_options(retries=2, retry_delay_seconds=1)(
                        "calculate_neighbors",
                        task_definition_arn,
                        context.user,
                        context.cluster,
                        context.security_groups.split(":"),
                        context.subnets.split(":"),
                        calculate_neighbors_command,
                    )
                else:
                    print(" ".join(calculate_neighbors_command))
