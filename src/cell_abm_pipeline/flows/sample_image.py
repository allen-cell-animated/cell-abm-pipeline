"""
Workflow for sampling cell ids from an image.

Working location structure:

.. code-block:: bash

    (name)
    ├── images
    │    └── (name)_(key).ome.tiff
    ├── plots
    │    └── plots.SAMPLE
    │        └── (name)_(key)_channel_(channel).SAMPLE.png
    └── samples
        └── samples.RAW
            └── (name)_(key)_channel_(channel).RAW.csv

The **images** directory contains the input image to be sampled.
Resulting sample(s) are placed into the **samples/samples.RAW** directory and
corresponding plot(s) are placed into the **plots/plots.SAMPLE** directory.
"""

from dataclasses import dataclass, field
from prefect import flow

from io_collection.keys import make_key
from io_collection.load import load_image
from io_collection.save import save_dataframe, save_figure
from abm_initialization_collection.image import get_image_bounds, plot_contact_sheet
from abm_initialization_collection.sample import (
    get_sample_indices,
    get_image_samples,
    scale_sample_coordinates,
)


# Pixel resolution for images (um/pixel) in x/y
SCALE_MICRONS_XY: float = 0.108333

# Pixel resolution for images (um/pixel) in z
SCALE_MICRONS_Z: float = 0.29


@dataclass
class ParametersConfig:
    key: str

    channels: list[int] = field(default_factory=lambda: [0])

    grid: str = "rect"

    coordinate_type: str = "index"

    resolution: float = 1.0

    scale_xy: float = SCALE_MICRONS_XY

    scale_z: float = SCALE_MICRONS_Z

    contact_sheet: bool = True

    extension: str = ".ome.tiff"


@dataclass
class ContextConfig:
    working_location: str


@dataclass
class SeriesConfig:
    name: str


@flow(name="sample-image")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    image_key = make_key(series.name, "images", f"{series.name}_{parameters.key}{parameters.extension}")
    image = load_image(
        context.working_location,
        image_key,
        dim_order="ZYX" if parameters.extension == ".tiff" else None,
    )
    image_bounds = get_image_bounds(image)

    sample_indices = get_sample_indices(
        parameters.grid,
        image_bounds,
        parameters.resolution,
        parameters.scale_xy,
        parameters.scale_z,
    )

    for channel in parameters.channels:
        channel_key = f"{series.name}_{parameters.key}_channel_{channel}"
        samples = get_image_samples(image, sample_indices, channel)
        samples = scale_sample_coordinates(
            samples,
            parameters.coordinate_type,
            parameters.resolution,
            parameters.scale_xy,
            parameters.scale_z,
        )
        sample_key = make_key(series.name, "samples", "samples.RAW", f"{channel_key}.RAW.csv")
        save_dataframe(context.working_location, sample_key, samples, index=False)

        if parameters.contact_sheet:
            contact_sheet = plot_contact_sheet(samples)
            plot_key = make_key(series.name, "plots", "plots.SAMPLE", f"{channel_key}.SAMPLE.png")
            save_figure(context.working_location, plot_key, contact_sheet)
