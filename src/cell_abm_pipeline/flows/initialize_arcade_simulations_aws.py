from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from arcade_collection.input import (
    convert_to_cells_file,
    convert_to_locations_file,
    generate_setup_file,
    merge_region_samples,
)
from container_collection.docker import run_docker_command
from io_collection.keys import change_key, check_key, make_key
from io_collection.load import load_dataframe
from io_collection.save import save_json, save_text
from prefect import flow, get_run_logger

from cell_abm_pipeline.flows.initialize_voronoi_simulations import (
    CRITICAL_HEIGHTS,
    CRITICAL_VOLUMES,
    HEIGHTS,
    POTTS_TERMS,
    STATE_THRESHOLDS,
    VOLUMES,
)


@dataclass
class ParametersConfig:
    image: str

    channels: dict

    margins: Tuple[int, int, int] = (0, 0, 0)

    volumes: dict = field(default_factory=lambda: VOLUMES)

    heights: dict = field(default_factory=lambda: HEIGHTS)

    critical_volumes: dict = field(default_factory=lambda: CRITICAL_VOLUMES)

    critical_heights: dict = field(default_factory=lambda: CRITICAL_HEIGHTS)

    state_thresholds: Dict[str, float] = field(default_factory=lambda: STATE_THRESHOLDS)

    potts_terms: list[str] = field(default_factory=lambda: POTTS_TERMS)


@dataclass
class ContextConfig:
    working_location: str

    reference_location: str

    access_key_id: Optional[str] = None

    secret_access_key: Optional[str] = None


@dataclass
class SeriesConfig:
    name: str

    reference_key: str

    conditions: list


@flow(name="initialize-arcade-simulations-aws")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    if not context.working_location.startswith("s3://"):
        logger = get_run_logger()
        logger.error("AWS ARCADE simulations must be initialized with S3 working location.")
        return

    if check_key(context.reference_location, series.reference_key):
        reference = load_dataframe(context.reference_location, series.reference_key)
    else:
        reference = None

    environment = []

    if context.access_key_id is not None:
        environment.append(f"AWS_ACCESS_KEY_ID={context.access_key_id}")

    if context.secret_access_key is not None:
        environment.append(f"AWS_SECRET_ACCESS_KEY={context.secret_access_key}")

    for fov in series.conditions:
        sample_images(
            series.name,
            fov["key"],
            context.working_location,
            parameters.image,
            environment,
            parameters.channels.keys(),
        )

        samples = {}
        for channel_index, channel_name in parameters.channels.items():
            process_samples(
                series.name,
                fov["key"],
                context.working_location,
                parameters.image,
                environment,
                channel_index,
                fov["include_ids"],
            )

            sample_key = rename_samples(
                series.name, fov["key"], context.working_location, channel_index, channel_name
            )
            channel_samples = load_dataframe(context.working_location, sample_key)
            samples[channel_name] = channel_samples

        merged_samples = merge_region_samples(samples, parameters.margins)

        cells = convert_to_cells_file(
            merged_samples,
            reference[reference["KEY"] == fov["key"]],
            parameters.volumes,
            parameters.heights,
            parameters.critical_volumes,
            parameters.critical_heights,
            parameters.state_thresholds,
        )
        cells_key = make_key(
            series.name, "converted", "converted.ARCADE", f"{series.name}_{fov['key']}.CELLS.json"
        )
        save_json(context.working_location, cells_key, cells)

        locations = convert_to_locations_file(merged_samples)
        locations_key = make_key(
            series.name,
            "converted",
            "converted.ARCADE",
            f"{series.name}_{fov['key']}.LOCATIONS.json",
        )
        save_json(context.working_location, locations_key, locations)

        setup = generate_setup_file(merged_samples, parameters.margins, parameters.potts_terms)
        setup_key = make_key(
            series.name, "converted", "converted.ARCADE", f"{series.name}_{fov['key']}.xml"
        )
        save_text(context.working_location, setup_key, setup)


@flow(name="sample-images")
def sample_images(name, key, working_location, image, environment, channels):
    sample_image_command = [
        "abmpipe",
        "sample-image",
        "::",
        f"parameters.key={key}",
        f"parameters.channels=[{','.join(channels)}]",
        "parameters.resolution=1.0",
        "parameters.grid=rect",
        "parameters.coordinate_type=step",
        f"context.working_location={working_location}",
        f"series.name={name}",
    ]
    run_docker_command(image, sample_image_command, environment=environment)


@flow(name="process-samples")
def process_samples(name, key, working_location, image, environment, channel_index, include_ids):
    process_samples_command = [
        "abmpipe",
        "process-sample",
        "::",
        f"parameters.key={key}",
        f"parameters.channel={channel_index}",
        "parameters.remove_unconnected=True",
        "parameters.unconnected_filter=connectivity",
        "parameters.remove_edges=False",
        f"parameters.include_ids=[{','.join([str(id) for id in include_ids])}]",
        f"context.working_location={working_location}",
        f"series.name={name}",
    ]
    run_docker_command(image, process_samples_command, environment=environment)


@flow(name="rename-samples")
def rename_samples(name, key, working_location, channel_index, channel_name):
    old_key = make_key(
        name,
        "samples",
        "samples.PROCESSED",
        f"{name}_{key}_channel_{channel_index}.PROCESSED.csv",
    )
    new_key = make_key(
        name,
        "samples",
        "samples.PROCESSED",
        f"{name}_{key}.PROCESSED.{channel_name}.csv",
    )

    change_key(working_location, old_key, new_key)

    return new_key
