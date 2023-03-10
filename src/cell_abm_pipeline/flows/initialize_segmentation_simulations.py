from dataclasses import dataclass, field
from typing import Optional

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

SUFFIXES: dict[str, str] = {"DEFAULT": "_cell", "NUCLEUS": "_nucleus"}

EXTENSIONS: dict[str, str] = {"DEFAULT": ".tiff", "NUCLEUS": ".tiff"}

VOLUMES: dict[str, tuple[float, float]] = {
    "DEFAULT": (1865.0, 517.0),
    "NUCLEUS": (542.0, 157.0),
}

HEIGHTS: dict[str, tuple[float, float]] = {
    "DEFAULT": (9.65, 2.4),
    "NUCLEUS": (6.75, 1.7),
}

CRITICAL_VOLUMES: dict[str, tuple[float, float]] = {
    "DEFAULT": (1300.0, 200.0),
    "NUCLEUS": (400.0, 50.0),
}

CRITICAL_HEIGHTS: dict[str, tuple[float, float]] = {
    "DEFAULT": (9.0, 2.0),
    "NUCLEUS": (6.5, 1.5),
}

STATE_THRESHOLDS: dict[str, float] = {
    "APOPTOTIC_LATE": 0.25,
    "APOPTOTIC_EARLY": 1,
    "PROLIFERATIVE_G1": 1.124,
    "PROLIFERATIVE_S": 1.726,
    "PROLIFERATIVE_G2": 1.969,
    "PROLIFERATIVE_M": 2,
}

POTTS_TERMS: list[str] = [
    "volume",
    "surface",
    "adhesion",
    "height",
    "junction",
    "substrate",
    "persistence",
]


@dataclass
class ParametersConfig:
    image: str

    suffixes: dict[str, str] = field(default_factory=lambda: SUFFIXES)

    extensions: dict[str, str] = field(default_factory=lambda: EXTENSIONS)

    volumes: dict = field(default_factory=lambda: VOLUMES)

    heights: dict = field(default_factory=lambda: HEIGHTS)

    critical_volumes: dict = field(default_factory=lambda: CRITICAL_VOLUMES)

    critical_heights: dict = field(default_factory=lambda: CRITICAL_HEIGHTS)

    state_thresholds: dict[str, float] = field(default_factory=lambda: STATE_THRESHOLDS)

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


@flow(name="initialize-segmentation-simulations")
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

    cell_suffix = parameters.suffixes["DEFAULT"]
    nuc_suffix = parameters.suffixes["NUCLEUS"]

    cell_extension = parameters.extensions["DEFAULT"]
    nuc_extension = parameters.extensions["NUCLEUS"]

    docker_volume = create_docker_volume(context.working_location)

    for fov in series.conditions:
        fov_key = fov["key"]
        cell_fov_key = f"{fov_key}{cell_suffix}"
        nuc_fov_key = f"{fov_key}{nuc_suffix}"

        # Sample cell and nucleus images
        sample_nuc_command = get_sample_command(series.name, nuc_fov_key, nuc_extension)
        run_docker_command(parameters.image, sample_nuc_command, volume=docker_volume)
        sample_cell_command = get_sample_command(series.name, cell_fov_key, cell_extension)
        run_docker_command(parameters.image, sample_cell_command, volume=docker_volume)

        # Process cell and nucleus samples
        process_nuc_command = get_process_command(series.name, nuc_fov_key)
        run_docker_command(parameters.image, process_nuc_command, volume=docker_volume)
        process_cell_command = get_process_command(series.name, cell_fov_key, fov["exclude_ids"])
        run_docker_command(parameters.image, process_cell_command, volume=docker_volume)

        # Rename processed nucleus and voronoi samples
        old_nuc_key = make_key(samples_key, f"{series.name}_{nuc_fov_key}_C00.PROCESSED.csv")
        new_nuc_key = make_key(samples_key, f"{series.name}_{fov_key}.PROCESSED.NUCLEUS.csv")
        change_key(context.working_location, old_nuc_key, new_nuc_key)

        old_cell_key = make_key(samples_key, f"{series.name}_{cell_fov_key}_C00.PROCESSED.csv")
        new_cell_key = make_key(samples_key, f"{series.name}_{fov_key}.PROCESSED.DEFAULT.csv")
        change_key(context.working_location, old_cell_key, new_cell_key)

        # Merge samples between regions
        samples = {
            "DEFAULT": load_dataframe(context.working_location, new_cell_key),
            "NUCLEUS": load_dataframe(context.working_location, new_nuc_key),
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


def get_sample_command(name: str, key: str, extension: str) -> list[str]:
    return [
        "abmpipe",
        "sample-image",
        "::",
        "parameters.channels=[0]",
        "parameters.resolution=1.0",
        "parameters.grid=rect",
        "parameters.coordinate_type=step",
        "context.working_location=/mnt",
        f"parameters.extension={extension}",
        f"series.name={name}",
        f"parameters.key={key}",
    ]


def get_process_command(name: str, key: str, ids: Optional[list[int]] = None) -> list[str]:
    return [
        "abmpipe",
        "process-sample",
        "::",
        "parameters.channel=0",
        "parameters.remove_unconnected=True",
        "parameters.unconnected_filter=connectivity",
        "context.working_location=/mnt",
        f"parameters.remove_edges={ids is not None}",
        f"parameters.edge_threshold={100 if ids is not None else 1}",
        f"series.name={name}",
        f"parameters.key={key}",
        f"parameters.exclude_ids=[{','.join([str(id) for id in ids]) if ids is not None else ''}]",
    ]
