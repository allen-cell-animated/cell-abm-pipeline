"""
Workflow for generating Voronoi tessellation images.
"""

from dataclasses import dataclass

from abm_initialization_collection.image import create_voronoi_image
from io_collection.keys import make_key
from io_collection.load import load_image
from io_collection.save import save_image
from prefect import flow


@dataclass
class ParametersConfig:
    """Parameter configuration for generate voronoi tessellation flow."""

    channel: int

    iterations: int

    target_height: int


@dataclass
class ContextConfig:
    """Context configuration for generate voronoi tessellation flow."""

    working_location: str


@dataclass
class SeriesConfig:
    """Series configuration for generate voronoi tessellation flow."""

    name: str

    conditions: list[dict]


@flow(name="generate-voronoi-tessellation")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    for key in series.conditions:
        image_key = make_key(series.name, "images", f"{series.name}_{key['key']}.tiff")
        image = load_image(context.working_location, image_key, "ZYX")

        voronoi = create_voronoi_image(
            image, parameters.channel, parameters.iterations, parameters.target_height
        )

        voronoi_key = make_key(
            series.name,
            "images",
            f"{series.name}_{key['key']}_C{parameters.channel:02}_voronoi.ome.tiff",
        )
        save_image(context.working_location, voronoi_key, voronoi)
