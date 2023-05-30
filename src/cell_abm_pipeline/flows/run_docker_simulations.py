"""
Workflow to run containers using local Docker.
"""

from dataclasses import dataclass, field
from typing import Optional, Union

from arcade_collection.input import group_template_conditions
from container_collection.docker import (
    check_docker_job,
    clean_docker_job,
    create_docker_volume,
    get_docker_logs,
    make_docker_job,
    remove_docker_volume,
    submit_docker_job,
    terminate_docker_job,
)
from container_collection.manifest import find_missing_conditions
from container_collection.template import generate_input_contents
from io_collection.keys import copy_key, make_key
from io_collection.load import load_dataframe, load_text
from io_collection.save import save_text
from prefect import flow, get_run_logger
from prefect.server.schemas.states import State

from cell_abm_pipeline.tasks.physicell import render_physicell_template


@dataclass
class ParametersConfig:
    model: str

    image: str

    retries: int

    retry_delay: int

    seeds_per_job: int = 1

    log_filter: str = ""

    terminate_jobs: bool = True

    save_logs: bool = True

    clean_jobs: bool = True


@dataclass
class ContextConfig:
    working_location: str

    manifest_location: str

    template_location: str


@dataclass
class SeriesConfig:
    name: str

    manifest_key: str

    template_key: str

    seeds: list[int]

    conditions: list[dict]

    extensions: list[str]

    inits: list[dict] = field(default_factory=lambda: [])

    groups: list[Optional[str]] = field(default_factory=lambda: [None])


@flow(name="run-docker-simulations")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    if context.working_location.startswith("s3://"):
        logger = get_run_logger()
        logger.error("Docker simulations can only be run with local working location.")
        return

    manifest = load_dataframe(context.manifest_location, series.manifest_key)
    template = load_text(context.template_location, series.template_key)

    job_key = make_key(context.working_location, series.name, "{{timestamp}}")
    volume = create_docker_volume(job_key)

    all_container_ids: list[str] = []

    for group in series.groups:
        group_key = series.name if group is None else f"{series.name}_{group}"
        group_conditions = [
            condition
            for condition in series.conditions
            if group is None or condition["group"] == group
        ]
        group_inits = [init for init in series.inits if group is None or init["group"] == group]

        # Find missing conditions.
        missing_conditions = find_missing_conditions(
            manifest, series.name, group_conditions, series.seeds, series.extensions
        )

        if len(missing_conditions) == 0:
            continue

        # Convert missing conditions into model input files.
        input_contents: list[str] = []

        if parameters.model.upper() == "ARCADE":
            condition_sets = group_template_conditions(missing_conditions, parameters.seeds_per_job)
            input_contents = generate_input_contents(template, condition_sets)
        elif parameters.model.upper() == "PHYSICELL":
            input_contents = render_physicell_template(template, missing_conditions, group_key)

        if len(input_contents) == 0:
            continue

        # Copy source init files to target init files.
        valid_seeds = {condition["seed"] for condition in missing_conditions}
        for init in group_inits:
            if len(valid_seeds.intersection(init["seeds"])) == 0:
                continue

            source_key = make_key(init["name"], "inits", f"inits.{parameters.model.upper()}")
            source = make_key(source_key, f"{init['name']}_{init['key']}")

            target_key = make_key(series.name, "{{timestamp}}", "inits")
            targets = [make_key(target_key, f"{group_key}_{seed:04d}") for seed in init["seeds"]]

            for target in targets:
                for ext in init["extensions"]:
                    copy_key(context.working_location, f"{source}.{ext}", f"{target}.{ext}")

        # Save input files and run jobs.
        for index, input_content in enumerate(input_contents):
            input_key = make_key(series.name, "{{timestamp}}", "inputs", f"{group_key}_{index}.xml")
            save_text(context.working_location, input_key, input_content)

            job_definition = make_docker_job(group_key, parameters.image, index)
            container_id = submit_docker_job(job_definition, volume)
            all_container_ids.append(container_id)

    all_jobs: list[Union[int, State]] = []

    for container_id in all_container_ids:
        exitcode = check_docker_job.with_options(
            retries=parameters.retries, retry_delay_seconds=parameters.retry_delay
        ).submit(container_id, parameters.retries)

        wait_for = [exitcode]

        if parameters.terminate_jobs:
            terminate_status = terminate_docker_job.submit(container_id, wait_for=wait_for)
            wait_for = [terminate_status]

        if parameters.save_logs:
            logs = get_docker_logs.submit(container_id, parameters.log_filter, wait_for=wait_for)
            log_key = make_key(series.name, "{{timestamp}}", "logs", f"{container_id}.log")
            save_text.submit(context.working_location, log_key, logs)
            wait_for = [logs]

        if parameters.clean_jobs:
            clean = clean_docker_job.submit(container_id, wait_for=wait_for)
            wait_for = [clean]

        all_jobs = all_jobs + wait_for

    remove_docker_volume.submit(volume, wait_for=all_jobs)
