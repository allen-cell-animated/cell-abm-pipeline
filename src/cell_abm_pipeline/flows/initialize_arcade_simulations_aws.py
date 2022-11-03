from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional
from prefect import flow, get_run_logger

from io_collection.keys import make_key, check_key, change_key
from io_collection.load import load_dataframe
from io_collection.save import save_text, save_json
from container_collection.docker import run_docker_command
from arcade_collection.input import (
    merge_region_samples,
    convert_to_cells_file,
    convert_to_locations_file,
    generate_setup_file,
)

from cell_abm_pipeline.flows.initialize_arcade_simulations_local import (
    VOLUME_DISTRIBUTIONS,
    HEIGHT_DISTRIBUTIONS,
    CRITICAL_VOLUME_DISTRIBUTIONS,
    CRITICAL_HEIGHT_DISTRIBUTIONS,
    STATE_THRESHOLDS,
    POTTS_TERMS,
)


@dataclass
class ParametersConfig:
    image: str

    channels: list

    margins: Tuple[int, int, int] = (0, 0, 0)

    volume_distributions: dict = field(default_factory=lambda: VOLUME_DISTRIBUTIONS)

    height_distributions: dict = field(default_factory=lambda: HEIGHT_DISTRIBUTIONS)

    critical_volume_distributions: dict = field(
        default_factory=lambda: CRITICAL_VOLUME_DISTRIBUTIONS
    )

    critical_height_distributions: dict = field(
        default_factory=lambda: CRITICAL_HEIGHT_DISTRIBUTIONS
    )

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

    channel_indices = [str(channel["index"]) for channel in parameters.channels]
    wait_for_fovs = []

    for fov in series.conditions:
        sample_image_command = [
            "abmpipe",
            "sample-image",
            "::",
            f"parameters.key={series.name}_{fov['key']}",
            f"parameters.channels=[{','.join(channel_indices)}]",
            "parameters.resolution=1.0",
            "parameters.grid=rect",
            "parameters.coordinate_type=step",
            f"context.working_location={context.working_location}",
            f"series.name={series.name}",
        ]
        sample_image = run_docker_command.submit(
            parameters.image, sample_image_command, environment=environment
        )

        samples = {}
        wait_for_samples = []

        for channel in parameters.channels:
            cell_ids = [str(cell_id) for cell_id in fov["cell_ids"]]

            process_samples_command = [
                "abmpipe",
                "process-sample",
                "::",
                f"parameters.key={series.name}_{fov['key']}",
                f"parameters.channel={channel['index']}",
                "parameters.remove_unconnected=True",
                "parameters.unconnected_filter=connectivity",
                "parameters.remove_edges=False",
                f"parameters.include_ids=[{','.join(cell_ids)}]",
                f"context.working_location={context.working_location}",
                f"series.name={series.name}",
            ]
            process_samples = run_docker_command.submit(
                parameters.image,
                process_samples_command,
                environment=environment,
                wait_for=[sample_image],
            )
            wait_for_fovs.append(process_samples)

            old_key = make_key(
                series.name,
                "samples",
                "samples.PROCESSED",
                f"{series.name}_{fov['key']}_channel_{channel['index']}.PROCESSED.csv",
            )
            new_key = make_key(
                series.name,
                "samples",
                "samples.PROCESSED",
                f"{series.name}_{fov['key']}.PROCESSED.{channel['name']}.csv",
            )
            rename = change_key.submit(
                context.working_location, old_key, new_key, wait_for=[process_samples]
            )

            channel_samples = load_dataframe.submit(
                context.working_location, new_key, wait_for=[rename]
            )
            samples[channel["name"]] = channel_samples
            wait_for_samples.append(channel_samples)

        merged_samples = merge_region_samples.submit(
            samples, parameters.margins, wait_for=wait_for_samples
        )

        cells = convert_to_cells_file.submit(
            merged_samples,
            reference[reference["KEY"] == fov["key"]],
            parameters.volume_distributions,
            parameters.height_distributions,
            parameters.critical_volume_distributions,
            parameters.critical_height_distributions,
            parameters.state_thresholds,
        )
        cells_key = make_key(
            series.name, "inits", "inits.ARCADE", f"{series.name}_{fov['key']}.CELLS.json"
        )
        save_json.submit(context.working_location, cells_key, cells)

        locations = convert_to_locations_file.submit(merged_samples)
        locations_key = make_key(
            series.name, "inits", "inits.ARCADE", f"{series.name}_{fov['key']}.LOCATIONS.json"
        )
        save_json.submit(context.working_location, locations_key, locations)

        setup = generate_setup_file.submit(
            merged_samples, parameters.margins, parameters.potts_terms
        )
        setup_key = make_key(
            series.name, "inits", "inits.ARCADE", f"{series.name}_{fov['key']}.xml"
        )
        save_text.submit(context.working_location, setup_key, setup)
