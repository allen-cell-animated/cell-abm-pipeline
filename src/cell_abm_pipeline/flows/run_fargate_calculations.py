"""
Workflow for running containerized calculations using Fargate.
"""

import importlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd
from container_collection.fargate import (
    make_fargate_task,
    register_fargate_task,
    submit_fargate_task,
)
from io_collection.keys import check_key, make_key, remove_key
from io_collection.load import load_dataframe
from io_collection.save import save_dataframe
from prefect import flow, get_run_logger

from cell_abm_pipeline.__config__ import make_dotlist_from_config


class Calculation(Enum):
    """Registered calculation types."""

    COEFFICIENTS = ("calculate_coefficients", True)

    NEIGHBORS = ("calculate_neighbors", False)

    POSITIONS = ("calculate_positions", False)

    PROPERTIES = ("calculate_properties", True)


@dataclass
class ParametersConfig:
    """Parameter configuration for run fargate calculations flow."""

    image: str

    ticks: list[int]

    calculate: Optional[Calculation] = None

    chunk: Optional[int] = None

    submit_tasks: bool = True

    overrides: dict = field(default_factory=lambda: {})


@dataclass
class ContextConfig:
    """Context configuration for run fargate calculations flow."""

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
    """Series configuration for run fargate calculations flow."""

    name: str

    seeds: list[int]

    conditions: list[dict]


@flow(name="run-fargate-calculations")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    """Main run fargate calculations flow."""

    # Check that a valid calculation type is selected.
    if parameters.calculate is None:
        logger = get_run_logger()
        logger.error(
            "No valid calculation type selected. Valid options: [ %s ]",
            " | ".join([member.name for member in Calculation]),
        )
        return

    # Check that the working location is vali.d
    if not context.working_location.startswith("s3://"):
        logger = get_run_logger()
        logger.error("Fargate calculations can only be run with S3 working location.")
        return

    # Get the calculation type.
    suffix = parameters.calculate.name
    module_name, chunkable = parameters.calculate.value
    analysis_key = make_key(series.name, "analysis", f"analysis.{suffix}")

    # Get the region suffix, if it exists.
    region = ""
    if "region" in parameters.overrides is not None:
        region = f"_{parameters.overrides['region']}"

    # Create and register the task definition for the calculation.
    task_definition = make_fargate_task(
        module_name,
        parameters.image,
        context.account,
        context.region,
        context.user,
        context.vcpus,
        context.memory,
    )
    task_definition_arn = register_fargate_task(task_definition)

    # Import the module for the specified calculation.
    module = importlib.import_module(f"..{module_name}", package=__name__)

    # Create the context and series configs for the calculation.
    context_config = module.ContextConfig(working_location=context.working_location)
    series_config = module.SeriesConfig(name=series.name)

    for condition in series.conditions:
        for seed in series.seeds:
            series_key = f"{series.name}_{condition['key']}_{seed:04d}"

            # If the calculation is chunkable, load results to identify chunks.
            if chunkable:
                results_key = make_key(series.name, "results", f"{series_key}.csv")
                results = load_dataframe(context.working_location, results_key)

            # Check if the compiled calculation result already exists.
            calc_key = make_key(analysis_key, f"{series_key}{region}.{suffix}.csv")
            calc_key_exists = check_key(context.working_location, calc_key)

            # If the compiled calculation result already exists, load the calculated ticks.
            existing_ticks = []
            if calc_key_exists:
                existing_calc = load_dataframe(context.working_location, calc_key, usecols=["TICK"])
                existing_ticks = list(existing_calc["TICK"].unique())

            for tick in parameters.ticks:
                # Skip the tick if it exists in the compiled calculation result.
                if tick in existing_ticks:
                    continue

                # Check if the individual calculation result already exists.
                tick_key = make_key(analysis_key, f"{series_key}_{tick:06d}{region}.{suffix}.csv")
                tick_key_exists = check_key(context.working_location, tick_key)

                # Skip the tick if the individual calculation result exists.
                if tick_key_exists:
                    continue

                completed_offset_keys = []
                missing_offsets_overrides = []

                # If the calculation is chunkable, get the completed and missing chunk offsets.
                if chunkable:
                    total = results[results["TICK"] == tick].shape[0]
                    chunk = parameters.chunk
                    all_offsets = list(range(0, total, chunk)) if chunk is not None else [0]

                    for offset in all_offsets:
                        if chunk is not None:
                            offset_key = tick_key.replace(
                                f".{suffix}.csv",
                                f".{offset:04d}.{chunk:04d}.{suffix}.csv",
                            )
                            offset_key_exists = check_key(context.working_location, offset_key)

                            if offset_key_exists:
                                completed_offset_keys.append(offset_key)
                                continue

                        missing_offsets_overrides.append({"offset": offset, "chunk": chunk})
                else:
                    missing_offsets_overrides.append({})

                # Create commands and submit (or display) tasks.
                for offset_overrides in missing_offsets_overrides:
                    parameters_config = module.ParametersConfig(
                        key=condition["key"],
                        seed=seed,
                        tick=tick,
                        **offset_overrides,
                        **parameters.overrides,
                    )

                    config = {
                        "context": context_config,
                        "series": series_config,
                        "parameters": parameters_config,
                    }

                    command = ["abmpipe", module_name, "::"] + make_dotlist_from_config(config)

                    if parameters.submit_tasks:
                        submit_fargate_task.with_options(retries=2, retry_delay_seconds=1)(
                            module_name,
                            task_definition_arn,
                            context.user,
                            context.cluster,
                            context.security_groups.split(":"),
                            context.subnets.split(":"),
                            command,
                        )
                    else:
                        print(" ".join(command))

                # If all chunk results exist, compile into unchunked result.
                if (
                    chunkable
                    and len(completed_offset_keys) == len(all_offsets)
                    and chunk is not None
                ):
                    tick_calcs = []

                    for key in completed_offset_keys:
                        tick_calcs.append(load_dataframe(context.working_location, key))

                    calc_dataframe = pd.concat(tick_calcs, ignore_index=True)
                    save_dataframe(context.working_location, tick_key, calc_dataframe, index=False)

                    for key in completed_offset_keys:
                        remove_key(context.working_location, key)
