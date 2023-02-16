from dataclasses import dataclass, field

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

    margins: tuple[int, int, int] = (0, 0, 0)

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

    samples_key = make_key(series.name, "samples", "samples.PROCESSED")
    converted_key = make_key(series.name, "converted", "converted.ARCADE")

    for fov in series.conditions:
        sample_image_command = [
            "abmpipe",
            "sample-image",
            "::",
            f"parameters.key={fov['key']}",
            f"parameters.channels=[{','.join(parameters.channels.keys())}]",
            "parameters.resolution=1.0",
            "parameters.grid=rect",
            "parameters.coordinate_type=step",
            "context.working_location=/mnt",
            f"series.name={series.name}",
        ]
        run_docker_command(parameters.image, sample_image_command, volume=volume)

        samples = {}
        for channel_index, channel_name in parameters.channels.items():
            process_samples_command = [
                "abmpipe",
                "process-sample",
                "::",
                f"parameters.key={fov['key']}",
                f"parameters.channel={channel_index}",
                "parameters.remove_unconnected=True",
                "parameters.unconnected_filter=connectivity",
                "parameters.remove_edges=False",
                f"parameters.include_ids=[{','.join([str(id) for id in fov['include_ids']])}]",
                "context.working_location=/mnt",
                f"series.name={series.name}",
            ]
            run_docker_command(parameters.image, process_samples_command, volume=volume)

            old_key = make_key(
                samples_key, f"{series.name}_{fov['key']}_channel_{channel_index}.PROCESSED.csv"
            )
            new_key = make_key(
                samples_key, f"{series.name}_{fov['key']}.PROCESSED.{channel_name}.csv"
            )
            change_key(context.working_location, old_key, new_key)

            channel_samples = load_dataframe(context.working_location, new_key)
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
        cells_key = make_key(converted_key, f"{series.name}_{fov['key']}.CELLS.json")
        save_json(context.working_location, cells_key, cells)

        locations = convert_to_locations_file(merged_samples)
        locations_key = make_key(converted_key, f"{series.name}_{fov['key']}.LOCATIONS.json")
        save_json(context.working_location, locations_key, locations)

        setup = generate_setup_file(merged_samples, parameters.margins, parameters.potts_terms)
        setup_key = make_key(converted_key, f"{series.name}_{fov['key']}.xml")
        save_text(context.working_location, setup_key, setup)

    remove_docker_volume(volume)
