"""
Workflow for running calculate shape properties flow across conditions and seeds.
"""

import copy
from dataclasses import dataclass

import pandas as pd
from container_collection.fargate import (
    make_fargate_task,
    register_fargate_task,
    submit_fargate_task,
)
from io_collection.keys import check_key, make_key, remove_key
from io_collection.load import load_dataframe
from io_collection.save import save_dataframe
from prefect import flow

from cell_abm_pipeline.__config__ import make_dotlist_from_config
from cell_abm_pipeline.flows.calculate_properties import (
    ContextConfig as CalculatePropertiesContextConfig,
)
from cell_abm_pipeline.flows.calculate_properties import (
    ParametersConfig as CalculatePropertiesParametersConfig,
)
from cell_abm_pipeline.flows.calculate_properties import (
    SeriesConfig as CalculatePropertiesSeriesConfig,
)

calculate_properties_COMMAND = ["abmpipe", "calculate-properties", "::"]


@dataclass
class ParametersConfig:
    """Parameter configuration for run calculate properties flow."""

    image: str

    frames: list[int]

    calculate_properties: CalculatePropertiesParametersConfig

    submit_tasks: bool = True


@dataclass
class ContextConfig:
    """Context configuration for run calculate properties flow."""

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
    """Series configuration for run calculate properties flow."""

    name: str

    seeds: list[int]

    conditions: list[dict]


@flow(name="run-calculate-properties")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    analysis_key = make_key(series.name, "analysis", "analysis.PROPS")

    region = ""
    if parameters.calculate_properties.region is not None:
        region = f"_{parameters.calculate_properties.region}"

    task_definition = make_fargate_task(
        "calculate_properties",
        parameters.image,
        context.account,
        context.region,
        context.user,
        context.vcpus,
        context.memory,
    )
    task_definition_arn = register_fargate_task(task_definition)

    context_config = CalculatePropertiesContextConfig(working_location=context.working_location)
    series_config = CalculatePropertiesSeriesConfig(name=series.name)

    for condition in series.conditions:
        for seed in series.seeds:
            series_key = f"{series.name}_{condition['key']}_{seed:04d}"

            results_key = make_key(series.name, "results", f"{series_key}.csv")
            results = load_dataframe(context.working_location, results_key)

            prop_key = make_key(analysis_key, f"{series_key}{region}.PROPS.csv")
            prop_key_exists = check_key(context.working_location, prop_key)

            existing_frames = []
            if prop_key_exists:
                existing_props = load_dataframe(
                    context.working_location, prop_key, usecols=["TICK"]
                )
                existing_frames = list(existing_props["TICK"].unique())

            for frame in parameters.frames:
                if frame in existing_frames:
                    continue

                frame_key = make_key(analysis_key, f"{series_key}_{frame:06d}{region}.PROPS.csv")
                frame_key_exists = check_key(context.working_location, frame_key)

                if frame_key_exists:
                    continue

                total = results[results["TICK"] == frame].shape[0]
                chunk = parameters.calculate_properties.chunk
                offsets = list(range(0, total, chunk)) if chunk is not None else [0]
                completed_keys = []

                for offset in offsets:
                    if chunk is not None:
                        offset_key = frame_key.replace(
                            ".PROPS.csv", f".{offset:04d}.{chunk:04d}.PROPS.csv"
                        )
                        offset_key_exists = check_key(context.working_location, offset_key)

                        if offset_key_exists:
                            completed_keys.append(offset_key)
                            continue

                    parameters_config = copy.deepcopy(parameters.calculate_properties)
                    parameters_config.key = parameters_config.key % condition["key"]
                    parameters_config.seed = seed
                    parameters_config.frame = frame
                    parameters_config.offset = offset

                    config = {
                        "context": context_config,
                        "series": series_config,
                        "parameters": parameters_config,
                    }

                    calculate_properties_command = (
                        calculate_properties_COMMAND + make_dotlist_from_config(config)
                    )

                    if parameters.submit_tasks:
                        submit_fargate_task.with_options(retries=2, retry_delay_seconds=1)(
                            "calculate_properties",
                            task_definition_arn,
                            context.user,
                            context.cluster,
                            context.security_groups.split(":"),
                            context.subnets.split(":"),
                            calculate_properties_command,
                        )
                    else:
                        print(" ".join(calculate_properties_command))

                if len(completed_keys) == len(offsets) and chunk is not None:
                    frame_props = []

                    for key in completed_keys:
                        frame_props.append(load_dataframe(context.working_location, key))

                    prop_dataframe = pd.concat(frame_props, ignore_index=True)
                    save_dataframe(context.working_location, frame_key, prop_dataframe, index=False)

                    for key in completed_keys:
                        remove_key(context.working_location, key)
