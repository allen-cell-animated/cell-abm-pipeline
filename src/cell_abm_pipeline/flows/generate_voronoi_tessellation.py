"""
Workflow for generating Voronoi tessellation images.

For each condition, a Voronoi tessellation is applied to the nuclear
segmentation to approximate the cell shape. The number of iterations and the
target height can be used to refine the tessellation to produce realistic cell
volumes.
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
    """Image channel."""

    iterations: int
    """Number of boundary estimation steps."""

    target_height: int
    """Target height in voxels."""


@dataclass
class ContextConfig:
    """Context configuration for generate voronoi tessellation flow."""

    working_location: str
    """Location for input and output files (local path or S3 bucket)."""


@dataclass
class SeriesConfig:
    """Series configuration for generate voronoi tessellation flow."""

    name: str
    """Name of the simulation series."""

    conditions: list[dict]
    """List of series condition dictionaries (must include unique condition "key")."""


@flow(name="generate-voronoi-tessellation")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    """Main generate voronoi tessellation flow."""

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
