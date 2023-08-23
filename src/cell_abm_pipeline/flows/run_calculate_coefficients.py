"""
Workflow for running calculate spherical harmonics coefficients flow across conditions and seeds.
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
from cell_abm_pipeline.flows.calculate_coefficients import (
    ContextConfig as CalculateCoefficientsContextConfig,
)
from cell_abm_pipeline.flows.calculate_coefficients import (
    ParametersConfig as CalculateCoefficientsParametersConfig,
)
from cell_abm_pipeline.flows.calculate_coefficients import (
    SeriesConfig as CalculateCoefficientsSeriesConfig,
)

CALCULATE_COEFFICIENTS_COMMAND = ["abmpipe", "calculate-coefficients", "::"]


@dataclass
class ParametersConfig:
    """Parameter configuration for run calculate coefficients flow."""

    image: str

    ticks: list[int]

    calculate_coefficients: CalculateCoefficientsParametersConfig

    submit_tasks: bool = True


@dataclass
class ContextConfig:
    """Context configuration for run calculate coefficients flow."""

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
    """Series configuration for run calculate coefficients flow."""

    name: str

    seeds: list[int]

    conditions: list[dict]


@flow(name="run-calculate-coefficients")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    """Main run calculate coefficients flow."""

    analysis_key = make_key(series.name, "analysis", "analysis.COEFFS")

    region = ""
    if parameters.calculate_coefficients.region is not None:
        region = f"_{parameters.calculate_coefficients.region}"

    task_definition = make_fargate_task(
        "calculate_coefficients",
        parameters.image,
        context.account,
        context.region,
        context.user,
        context.vcpus,
        context.memory,
    )
    task_definition_arn = register_fargate_task(task_definition)

    context_config = CalculateCoefficientsContextConfig(working_location=context.working_location)
    series_config = CalculateCoefficientsSeriesConfig(name=series.name)

    for condition in series.conditions:
        for seed in series.seeds:
            series_key = f"{series.name}_{condition['key']}_{seed:04d}"

            results_key = make_key(series.name, "results", f"{series_key}.csv")
            results = load_dataframe(context.working_location, results_key)

            coeff_key = make_key(analysis_key, f"{series_key}{region}.COEFFS.csv")
            coeff_key_exists = check_key(context.working_location, coeff_key)

            existing_ticks = []
            if coeff_key_exists:
                existing_coeffs = load_dataframe(
                    context.working_location, coeff_key, usecols=["TICK"]
                )
                existing_ticks = list(existing_coeffs["TICK"].unique())

            for tick in parameters.ticks:
                if tick in existing_ticks:
                    continue

                tick_key = make_key(analysis_key, f"{series_key}_{tick:06d}{region}.COEFFS.csv")
                tick_key_exists = check_key(context.working_location, tick_key)

                if tick_key_exists:
                    continue

                total = results[results["TICK"] == tick].shape[0]
                chunk = parameters.calculate_coefficients.chunk
                offsets = list(range(0, total, chunk)) if chunk is not None else [0]
                completed_keys = []

                for offset in offsets:
                    if chunk is not None:
                        offset_key = tick_key.replace(
                            ".COEFFS.csv", f".{offset:04d}.{chunk:04d}.COEFFS.csv"
                        )
                        offset_key_exists = check_key(context.working_location, offset_key)

                        if offset_key_exists:
                            completed_keys.append(offset_key)
                            continue

                    parameters_config = copy.deepcopy(parameters.calculate_coefficients)
                    parameters_config.key = parameters_config.key % condition["key"]
                    parameters_config.seed = seed
                    parameters_config.tick = tick
                    parameters_config.offset = offset

                    config = {
                        "context": context_config,
                        "series": series_config,
                        "parameters": parameters_config,
                    }

                    calculate_coefficients_command = (
                        CALCULATE_COEFFICIENTS_COMMAND + make_dotlist_from_config(config)
                    )

                    if parameters.submit_tasks:
                        submit_fargate_task.with_options(retries=2, retry_delay_seconds=1)(
                            "calculate_coefficients",
                            task_definition_arn,
                            context.user,
                            context.cluster,
                            context.security_groups.split(":"),
                            context.subnets.split(":"),
                            calculate_coefficients_command,
                        )
                    else:
                        print(" ".join(calculate_coefficients_command))

                if len(completed_keys) == len(offsets) and chunk is not None:
                    tick_coeffs = []

                    for key in completed_keys:
                        tick_coeffs.append(load_dataframe(context.working_location, key))

                    coeff_dataframe = pd.concat(tick_coeffs, ignore_index=True)
                    save_dataframe(context.working_location, tick_key, coeff_dataframe, index=False)

                    for key in completed_keys:
                        remove_key(context.working_location, key)
