"""
Workflow to run containers using AWS Batch.
"""

from dataclasses import dataclass, field
from typing import Optional

from arcade_collection.input import group_template_conditions
from container_collection.batch import (
    check_batch_job,
    get_batch_logs,
    make_batch_job,
    register_batch_job,
    submit_batch_job,
    terminate_batch_job,
)
from container_collection.manifest import find_missing_conditions
from container_collection.template import generate_input_contents
from io_collection.keys import copy_key, make_key
from io_collection.load import load_dataframe, load_text
from io_collection.save import save_text
from prefect import flow, get_run_logger


@dataclass
class ParametersConfig:
    seeds_per_job: int

    image: str

    retries: int

    retry_delay: int

    log_filter: str = ""

    terminate_jobs: bool = True

    save_logs: bool = True

    clean_jobs: bool = True


@dataclass
class ContextConfig:
    working_location: str

    manifest_location: str

    template_location: str

    account: str

    region: str

    user: str

    vcpus: int

    memory: int

    queue: str


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


@flow(name="run-batch-simulations")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    if not context.working_location.startswith("s3://"):
        logger = get_run_logger()
        logger.error("Batch simulations can only be run with S3 working location.")
        return

    manifest = load_dataframe(context.manifest_location, series.manifest_key)
    template = load_text(context.template_location, series.template_key)

    all_job_arns: list[str] = []

    for group in series.groups:
        group_key = series.name if group is None else f"{series.name}_{group}"
        group_conditions = [
            condition
            for condition in series.conditions
            if group is None or condition["group"] == group
        ]
        group_inits = [init for init in series.inits if group is None or init["group"] == group]

        # Find missing conditions and group into input file contents.
        missing_conditions = find_missing_conditions(
            manifest, series.name, group_conditions, series.seeds, series.extensions
        )

        if len(missing_conditions) == 0:
            continue

        condition_sets = group_template_conditions(missing_conditions, parameters.seeds_per_job)
        input_contents = generate_input_contents(template, condition_sets)

        # Copy source init files to target init files.
        valid_seeds = set([condition["seed"] for condition in missing_conditions])
        for init in group_inits:
            if len(valid_seeds.intersection(init["seeds"])) == 0:
                continue

            source_key = make_key(init["name"], "converted", "converted.ARCADE")
            source = make_key(source_key, f"{init['name']}_{init['key']}")

            target_key = make_key(series.name, "{{timestamp}}", "inits")
            targets = [make_key(target_key, f"{group_key}_{seed:04d}") for seed in init["seeds"]]

            for target in targets:
                for ext in ["CELLS.json", "LOCATIONS.json"]:
                    copy_key(context.working_location, f"{source}.{ext}", f"{target}.{ext}")

        # Create job definition.
        job_key = make_key(context.working_location, series.name, "{{timestamp}}/")
        job_definition = make_batch_job(
            group_key,
            parameters.image,
            context.account,
            context.region,
            context.user,
            context.vcpus,
            context.memory,
            job_key,
        )
        job_definition_arn = register_batch_job(job_definition)

        # Save input files.
        for index, input_content in enumerate(input_contents):
            input_key = make_key(series.name, "{{timestamp}}", "inputs", f"{group_key}_{index}.xml")
            save_text(context.working_location, input_key, input_content)

        # Submit jobs.
        job_arns = submit_batch_job(
            group_key,
            job_definition_arn,
            context.user,
            context.queue,
            len(input_contents),
        )
        all_job_arns = all_job_arns + job_arns

    for job_arn in all_job_arns:
        exitcode = check_batch_job.with_options(
            retries=parameters.retries, retry_delay_seconds=parameters.retry_delay
        ).submit(job_arn, parameters.retries)

        wait_for = [exitcode]

        if parameters.terminate_jobs:
            terminate_status = terminate_batch_job.submit(job_arn, wait_for=wait_for)
            wait_for = [terminate_status]

        if parameters.save_logs:
            logs = get_batch_logs.submit(job_arn, parameters.log_filter, wait_for=wait_for)
            log_key = make_key(series.name, "{{timestamp}}", "logs", f"{job_arn}.log")
            save_text.submit(context.working_location, log_key, logs)
            wait_for = [logs]
