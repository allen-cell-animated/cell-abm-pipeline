"""
Workflow for running containerized models using local Docker.

Working location structure:

.. code-block:: bash

    (name)
    └── YYYY-MM-DD
        ├── inits
        │   └── (name)_(group)_(seed).(extension)
        └── inputs
            └── (name)_(group)_(index).xml

The simulation series manifest, produced by the summarize manifest flow, is used
to identify which simulation conditions and seeds are missing. These conditions
and seeds are converted into input files using the given template file, grouped
by the specified job size. The relevant initialization and input files are then
saved to a dated directory. All simulations in the same group will use the same
initialization file for a given seed; if different initializations need to be
used for different conditions, assign the conditions to different groups.

Jobs are submitted to run via Docker using a volume to hold input and output
files. The jobs are periodically queried for status at the specified retry delay
interval, for the specified number of retries. If jobs are still running after
these retries are complete, the job is not terminated unless specified. Output
logs are also saved after these retries are complete. Note that if the job is
not complete when the logs are saved, only the logs available at that time will
be saved. The running containers and the mounted volume are removed unless
specified.

Note that this workflow works only if working location is local. For S3 working
locations, use the run batch simulations flow instead.
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
    """Parameter configuration for run docker simulations flow."""

    model: str
    """Name of model."""

    image: str
    """Name of model image."""

    retries: int
    """Number of retries to check if jobs are complete."""

    retry_delay: int
    """Delay between retries in seconds."""

    seeds_per_job: int = 1
    """Number of seeds per job."""

    log_filter: str = ""
    """Filter pattern for logs."""

    terminate_jobs: bool = True
    """True if jobs should be terminated after total retry time, False otherwise."""

    save_logs: bool = True
    """True to save job logs, False otherwise."""

    clean_jobs: bool = True
    """True to clean up job files, False otherwise."""


@dataclass
class ContextConfig:
    """Context configuration for run docker simulations flow."""

    working_location: str
    """Location for input and output files (local path or S3 bucket)."""

    manifest_location: str
    """Location of manifest file (local path or S3 bucket)."""

    template_location: str
    """Location of template file (local path or S3 bucket)."""


@dataclass
class SeriesConfig:
    """Series configuration for run docker simulations flow."""

    name: str
    """Name of the simulation series."""

    manifest_key: str
    """Key for manifest file."""

    template_key: str
    """Key for template file."""

    seeds: list[int]
    """List of series random seeds."""

    conditions: list[dict]
    """List of series condition dictionaries (must include unique condition "key")."""

    extensions: list[str]
    """List of file extensions in complete run."""

    inits: list[dict] = field(default_factory=lambda: [])
    """Initialization keys and associated group names."""

    groups: dict[str, Optional[str]] = field(default_factory=lambda: {"_": ""})
    """Initialization groups, keyed by group name."""


@flow(name="run-docker-simulations")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    """Main run docker simulations flow."""

    if context.working_location.startswith("s3://"):
        logger = get_run_logger()
        logger.error("Docker simulations can only be run with local working location.")
        return

    manifest = load_dataframe(context.manifest_location, series.manifest_key)
    template = load_text(context.template_location, series.template_key)

    job_key = make_key(context.working_location, series.name, "{{timestamp}}")
    volume = create_docker_volume(job_key)

    all_container_ids: list[str] = []

    for group in series.groups.keys():
        if series.groups[group] is None:
            continue

        group_key = series.name if group == "_" else f"{series.name}_{group}"
        group_conditions = [
            condition
            for condition in series.conditions
            if group is "_" or condition["group"] == group
        ]
        group_inits = [init for init in series.inits if group == "_" or init["group"] == group]

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
