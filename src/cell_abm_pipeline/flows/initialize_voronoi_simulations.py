from dataclasses import dataclass, field
from typing import Dict, Tuple

from arcade_collection.input import (
    convert_to_cells_file,
    convert_to_locations_file,
    generate_setup_file,
    merge_region_samples,
)
from container_collection.docker import (
    create_docker_volume,
    remove_docker_volume,
    run_docker_command,
)
from io_collection.keys import change_key, check_key, make_key
from io_collection.load import load_dataframe
from io_collection.save import save_json, save_text
from prefect import flow, get_run_logger

VOLUMES: Dict[str, Tuple[float, float]] = {
    "DEFAULT": (1865.0, 517.0),
    "NUCLEUS": (542.0, 157.0),
}

HEIGHTS: Dict[str, Tuple[float, float]] = {
    "DEFAULT": (9.65, 2.4),
    "NUCLEUS": (6.75, 1.7),
}

CRITICAL_VOLUMES: Dict[str, Tuple[float, float]] = {
    "DEFAULT": (1300.0, 200.0),
    "NUCLEUS": (400.0, 50.0),
}

CRITICAL_HEIGHTS: Dict[str, Tuple[float, float]] = {
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


@dataclass
class SeriesConfig:
    name: str

    reference_key: str

    conditions: list


SAMPLE_IMAGE_FLOW = ["abmpipe", "sample-image", "::"]

SAMPLE_IMAGE_DOTLIST = [
    "parameters.channels=[0]",
    "parameters.resolution=1.0",
    "parameters.grid=rect",
    "parameters.coordinate_type=step",
    "context.working_location=/mnt",
]

SAMPLE_IMAGE_DOTLIST_NUCLEUS = SAMPLE_IMAGE_DOTLIST + [
    "parameters.extension=.tiff",
]

PROCESS_SAMPLE_FLOW = ["abmpipe", "process-sample", "::"]

PROCESS_SAMPLE_DOTLIST = [
    "parameters.channel=0",
    "parameters.remove_unconnected=True",
    "parameters.unconnected_filter=connectivity",
    "context.working_location=/mnt",
]

PROCESS_SAMPLE_DOTLIST_NUCLEUS = PROCESS_SAMPLE_DOTLIST + [
    "parameters.remove_edges=False",
]

PROCESS_SAMPLE_DOTLIST_VORONOI = PROCESS_SAMPLE_DOTLIST + [
    "parameters.remove_edges=True",
    "parameters.edge_threshold=100",
]


@flow(name="initialize-voronoi-simulations")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    if context.working_location.startswith("s3://"):
        logger = get_run_logger()
        logger.error("Local ARCADE simulations must be initialized with local working location.")
        return

    if check_key(context.reference_location, series.reference_key):
        reference = load_dataframe(context.reference_location, series.reference_key)
    else:
        reference = None

    samples_key = make_key(series.name, "samples", "samples.PROCESSED")
    converted_key = make_key(series.name, "converted", "converted.ARCADE")

    docker_volume = create_docker_volume(context.working_location)

    for fov in series.conditions:
        fov_key = fov["key"]
        nucleus_key = [f"series.name={series.name}", f"parameters.key={fov_key}_T0000"]
        voronoi_key = [f"series.name={series.name}", f"parameters.key={fov_key}_T0000_C00_voronoi"]

        # Sample nucleus and voronoi images
        sample_nucleus_command = SAMPLE_IMAGE_FLOW + SAMPLE_IMAGE_DOTLIST_NUCLEUS + nucleus_key
        run_docker_command(parameters.image, sample_nucleus_command, volume=docker_volume)
        sample_voronoi_command = SAMPLE_IMAGE_FLOW + SAMPLE_IMAGE_DOTLIST + voronoi_key
        run_docker_command(parameters.image, sample_voronoi_command, volume=docker_volume)

        # Process nucleus and voronoi samples
        process_nucleus_command = PROCESS_SAMPLE_FLOW + PROCESS_SAMPLE_DOTLIST_NUCLEUS + nucleus_key
        run_docker_command(parameters.image, process_nucleus_command, volume=docker_volume)
        process_voronoi_command = (
            PROCESS_SAMPLE_FLOW
            + PROCESS_SAMPLE_DOTLIST_VORONOI
            + voronoi_key
            + [f"parameters.exclude_ids=[{','.join([str(id) for id in fov['exclude_ids']])}]"]
        )
        run_docker_command(parameters.image, process_voronoi_command, volume=docker_volume)

        # Rename processed nucleus and voronoi samples
        old_nucleus_key = make_key(
            samples_key, f"{series.name}_{fov_key}_T0000_channel_0.PROCESSED.csv"
        )
        new_nucleus_key = make_key(samples_key, f"{series.name}_{fov_key}.PROCESSED.NUCLEUS.csv")
        change_key(context.working_location, old_nucleus_key, new_nucleus_key)

        old_voronoi_key = make_key(
            samples_key, f"{series.name}_{fov_key}_T0000_C00_voronoi_channel_0.PROCESSED.csv"
        )
        new_default_key = make_key(samples_key, f"{series.name}_{fov_key}.PROCESSED.DEFAULT.csv")
        change_key(context.working_location, old_voronoi_key, new_default_key)

        # Merge samples between regions
        samples = {
            "DEFAULT": load_dataframe(context.working_location, new_default_key),
            "NUCLEUS": load_dataframe(context.working_location, new_nucleus_key),
        }
        merged_samples = merge_region_samples(samples, fov["margins"])

        # Convert samples to CELLS file
        cells = convert_to_cells_file(
            merged_samples,
            reference[reference["KEY"] == fov["key"]],
            parameters.volumes,
            parameters.heights,
            parameters.critical_volumes,
            parameters.critical_heights,
            parameters.state_thresholds,
        )
        cells_key = make_key(converted_key, f"{series.name}_{fov['key']}.CELLS.json")
        save_json(context.working_location, cells_key, cells)

        # Convert samples to LOCATIONS file
        locations = convert_to_locations_file(merged_samples)
        locations_key = make_key(converted_key, f"{series.name}_{fov['key']}.LOCATIONS.json")
        save_json(context.working_location, locations_key, locations)

        # Generate ARCADE setup file
        setup = generate_setup_file(merged_samples, fov["margins"], parameters.potts_terms)
        setup_key = make_key(converted_key, f"{series.name}_{fov['key']}.xml")
        save_text(context.working_location, setup_key, setup)

    remove_docker_volume(docker_volume)
