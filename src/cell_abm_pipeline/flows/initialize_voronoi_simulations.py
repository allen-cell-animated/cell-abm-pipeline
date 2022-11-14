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

# ==============================================================================


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


# ==============================================================================


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

    docker_volume = create_docker_volume(context.working_location)

    for fov in series.conditions:
        sample_nucleus_image(series.name, fov["key"], parameters.image, docker_volume)
        sample_voronoi_image(series.name, fov["key"], parameters.image, docker_volume)

        process_nucleus_samples(series.name, fov["key"], parameters.image, docker_volume)
        process_voronoi_samples(
            series.name, fov["key"], parameters.image, docker_volume, fov["exclude_ids"]
        )

        nucleus_sample_key = rename_nucleus_samples(
            series.name, fov["key"], context.working_location
        )
        default_sample_key = rename_voronoi_samples(
            series.name, fov["key"], context.working_location
        )

        samples = {
            "DEFAULT": load_dataframe(context.working_location, default_sample_key),
            "NUCLEUS": load_dataframe(context.working_location, nucleus_sample_key),
        }
        merged_samples = merge_region_samples(samples, fov["margins"])

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

        setup = generate_setup_file(merged_samples, fov["margins"], parameters.potts_terms)
        setup_key = make_key(
            series.name, "converted", "converted.ARCADE", f"{series.name}_{fov['key']}.xml"
        )
        save_text(context.working_location, setup_key, setup)

    remove_docker_volume(docker_volume)


# ==============================================================================


@flow(name="sample-nucleus-image")
def sample_nucleus_image(name, key, image, volume):
    sample_image_command = [
        "abmpipe",
        "sample-image",
        "::",
        f"parameters.key={name}_{key}_T0000",
        f"parameters.channels=[0]",
        "parameters.resolution=1.0",
        "parameters.grid=rect",
        "parameters.coordinate_type=step",
        "parameters.extension=.tiff",
        "context.working_location=/mnt",
        f"series.name={name}",
    ]
    sample_image = run_docker_command(image, sample_image_command, volume=volume)


@flow(name="sample-voronoi-image")
def sample_voronoi_image(name, key, image, volume):
    sample_image_command = [
        "abmpipe",
        "sample-image",
        "::",
        f"parameters.key={name}_{key}_T0000_C00_voronoi",
        f"parameters.channels=[0]",
        "parameters.resolution=1.0",
        "parameters.grid=rect",
        "parameters.coordinate_type=step",
        "context.working_location=/mnt",
        f"series.name={name}",
    ]
    sample_image = run_docker_command(image, sample_image_command, volume=volume)


@flow(name="process-nucleus-samples")
def process_nucleus_samples(name, key, image, volume):
    process_samples_command = [
        "abmpipe",
        "process-sample",
        "::",
        f"parameters.key={name}_{key}_T0000",
        f"parameters.channel=0",
        "parameters.remove_unconnected=True",
        "parameters.unconnected_filter=connectivity",
        "parameters.remove_edges=False",
        "context.working_location=/mnt",
        f"series.name={name}",
    ]
    process_samples = run_docker_command(image, process_samples_command, volume=volume)


@flow(name="process-voronoi-samples")
def process_voronoi_samples(name, key, image, volume, exclude_ids):
    process_samples_command = [
        "abmpipe",
        "process-sample",
        "::",
        f"parameters.key={name}_{key}_T0000_C00_voronoi",
        f"parameters.channel=0",
        "parameters.remove_unconnected=True",
        "parameters.unconnected_filter=connectivity",
        "parameters.remove_edges=True",
        f"parameters.exclude_ids=[{','.join([str(id) for id in exclude_ids])}]",
        "parameters.edge_threshold=100",
        "context.working_location=/mnt",
        f"series.name={name}",
    ]
    process_samples = run_docker_command(image, process_samples_command, volume=volume)


@flow(name="rename-nucleus-samples")
def rename_nucleus_samples(name, key, working_location):
    old_key = make_key(
        name,
        "samples",
        "samples.PROCESSED",
        f"{name}_{key}_T0000_channel_0.PROCESSED.csv",
    )
    new_key = make_key(
        name,
        "samples",
        "samples.PROCESSED",
        f"{name}_{key}.PROCESSED.NUCLEUS.csv",
    )

    change_key(working_location, old_key, new_key)

    return new_key


@flow(name="rename-voronoi-samples")
def rename_voronoi_samples(name, key, working_location):
    old_key = make_key(
        name,
        "samples",
        "samples.PROCESSED",
        f"{name}_{key}_T0000_C00_voronoi_channel_0.PROCESSED.csv",
    )
    new_key = make_key(
        name,
        "samples",
        "samples.PROCESSED",
        f"{name}_{key}.PROCESSED.DEFAULT.csv",
    )

    change_key(working_location, old_key, new_key)

    return new_key
