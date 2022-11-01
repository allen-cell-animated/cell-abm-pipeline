from dataclasses import dataclass
from typing import Optional
from prefect import flow, get_run_logger

from io_collection.keys import make_key, check_key, change_key
from io_collection.load import load_dataframe
from io_collection.quilt import load_quilt_package, save_quilt_item
from container_collection.docker import run_docker_command
from abm_initialization_collection.image import select_fov_images


@dataclass
class ParametersConfig:
    cells_per_fov: int

    num_fovs: int

    image: str

    quilt_package: str = "aics/hipsc_single_cell_image_dataset"

    quilt_registry: str = "s3://allencell"


@dataclass
class ContextConfig:
    working_location: str

    metadata_location: str

    access_key_id: Optional[str] = None

    secret_access_key: Optional[str] = None


@dataclass
class SeriesConfig:
    name: str

    metadata_key: str


@flow(name="initialize-arcade-simulations-aws")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    if not context.working_location.startswith("s3://"):
        logger = get_run_logger()
        logger.error("AWS ARCADE simulations can only be initialized with S3 working location.")
        return

    package = load_quilt_package(parameters.quilt_package, parameters.quilt_registry)
    key_exists = check_key(context.metadata_location, series.metadata_key)

    if not key_exists:
        save_quilt_item(context.metadata_location, series.metadata_key, package, "metadata.csv")

    metadata = load_dataframe(
        context.metadata_location,
        series.metadata_key,
        usecols=["CellId", "cell_stage", "outlier", "fov_seg_path", "this_cell_index"],
    )

    selected_fovs = select_fov_images(metadata, parameters.cells_per_fov, parameters.num_fovs)

    environment = []

    if context.access_key_id is not None:
        environment.append(f"AWS_ACCESS_KEY_ID={context.access_key_id}")

    if context.secret_access_key is not None:
        environment.append(f"AWS_SECRET_ACCESS_KEY={context.secret_access_key}")

    for fov in selected_fovs:
        fov_key = make_key(series.name, "images", f"{fov['key']}.ome.tiff")
        key_exists = check_key(context.working_location, fov_key)

        if not key_exists:
            save_quilt_item.submit(context.working_location, fov_key, package, fov["item"])

        sample_image_command = [
            "abmpipe",
            "sample-image",
            "::",
            f"parameters.key={fov['key']}",
            "parameters.channels=[0,1]",
            "parameters.resolution=1.0",
            "parameters.grid=rect",
            "parameters.coordinate_type=step",
            f"context.working_location={context.working_location}",
            f"series.name={series.name}",
        ]
        sample_image = run_docker_command.submit(
            parameters.image, sample_image_command, environment=environment
        )

        for channel in [0, 1]:
            process_samples_command = [
                "abmpipe",
                "process-sample",
                "::",
                f"parameters.key={fov['key']}",
                f"parameters.channel={channel}",
                "parameters.remove_unconnected=True",
                "parameters.unconnected_filter=connectivity",
                "parameters.remove_edges=False",
                f"parameters.include_ids=[{','.join([str(cell_id) for cell_id in fov['cell_ids']])}]",
                f"context.working_location={context.working_location}",
                f"series.name={series.name}",
            ]
            process_samples = run_docker_command.submit(
                parameters.image,
                process_samples_command,
                environment=environment,
                wait_for=[sample_image],
            )

            old_key = make_key(
                series.name,
                "samples",
                "samples.PROCESSED",
                f"{fov['key']}_channel_{channel}.PROCESSED.csv",
            )
            new_key = make_key(
                series.name,
                "samples",
                "samples.PROCESSED",
                f"{fov['key']}.PROCESSED{'.NUCLEUS' if channel == 0 else ''}.csv",
            )
            rename = change_key.submit(
                context.working_location, old_key, new_key, wait_for=[process_samples]
            )
