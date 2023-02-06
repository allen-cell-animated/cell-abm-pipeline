from dataclasses import dataclass

from abm_initialization_collection.image import select_fov_images
from io_collection.keys import check_key, make_key
from io_collection.load import load_dataframe
from io_collection.quilt import load_quilt_package, save_quilt_item
from prefect import flow


@dataclass
class ParametersConfig:
    cells_per_fov: int

    num_fovs: int

    quilt_package: str = "aics/hipsc_single_cell_image_dataset"

    quilt_registry: str = "s3://allencell"


@dataclass
class ContextConfig:
    working_location: str

    metadata_location: str


@dataclass
class SeriesConfig:
    name: str

    metadata_key: str


@flow(name="download-images")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
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

    for fov in selected_fovs:
        fov_key = make_key(series.name, "images", f"{series.name}_{fov['key']}.ome.tiff")
        key_exists = check_key(context.working_location, fov_key)

        if not key_exists:
            save_quilt_item.submit(context.working_location, fov_key, package, fov["item"])
