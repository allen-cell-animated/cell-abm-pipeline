from dataclasses import dataclass, field
from typing import Tuple, Dict
from prefect import flow, get_run_logger

from io_collection.keys import make_key, check_key, change_key
from io_collection.load import load_dataframe
from io_collection.save import save_text, save_json
from container_collection.docker import (
    create_docker_volume,
    remove_docker_volume,
    run_docker_command,
)
from arcade_collection.input import (
    merge_region_samples,
    convert_to_cells_file,
    convert_to_locations_file,
    generate_setup_file,
)


VOLUME_DISTRIBUTIONS: Dict[str, Tuple[float, float]] = {
    "DEFAULT": (1865.0, 517.0),
    "NUCLEUS": (542.0, 157.0),
}

HEIGHT_DISTRIBUTIONS: Dict[str, Tuple[float, float]] = {
    "DEFAULT": (9.65, 2.4),
    "NUCLEUS": (6.75, 1.7),
}

CRITICAL_VOLUME_DISTRIBUTIONS: Dict[str, Tuple[float, float]] = {
    "DEFAULT": (1300.0, 200.0),
    "NUCLEUS": (400.0, 50.0),
}

CRITICAL_HEIGHT_DISTRIBUTIONS: Dict[str, Tuple[float, float]] = {
    "DEFAULT": (9.0, 2.0),
    "NUCLEUS": (6.5, 1.5),
}

STATE_THRESHOLDS: Dict[str, float] = {
    "APOPTOTIC_LATE": 0.25,
    "APOPTOTIC_EARLY": 1,
    "PROLIFERATIVE_G1": 1.124,
    "PROLIFERATIVE_S": 1.726,
    "PROLIFERATIVE_G2": 1.969,
}

POTTS_TERMS: list[str] = [
    "volume",
    "surface",
    "adhesion",
    "height",
    "substrate",
    "persistence",
]


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


@dataclass
class SeriesConfig:
    name: str

    reference_key: str

    conditions: list


@flow(name="initialize-arcade-simulations-local")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    if context.working_location.startswith("s3://"):
        logger = get_run_logger()
        logger.error("Local ARCADE simulations must be initialized with local working location.")
        return

    if check_key(context.reference_location, series.reference_key):
        reference = load_dataframe(context.reference_location, series.reference_key)
    else:
        reference = None

    volume = create_docker_volume(context.working_location)
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
            "context.working_location=/mnt",
            f"series.name={series.name}",
        ]
        sample_image = run_docker_command.submit(
            parameters.image, sample_image_command, volume=volume
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
                "context.working_location=/mnt",
                f"series.name={series.name}",
            ]
            process_samples = run_docker_command.submit(
                parameters.image,
                process_samples_command,
                volume=volume,
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

    remove_docker_volume.submit(volume, wait_for=wait_for_fovs)
