"""
Workflow for converting ARCADE simulations to other formats.
"""

from dataclasses import dataclass, field
from typing import Optional

from arcade_collection.output import (
    convert_to_colorizer,
    convert_to_images,
    convert_to_meshes,
    convert_to_simularium,
)
from io_collection.keys import make_key
from io_collection.load import load_dataframe, load_tar
from io_collection.save import save_image, save_json, save_text
from prefect import flow

from cell_abm_pipeline.flows.plot_basic_metrics import PHASE_COLORS

FORMATS: list[str] = [
    "colorizer",
    "image",
    "mesh",
    "simularium",
]

FEATURES: list[str] = [
    "volume",
    "height",
]


@dataclass
class ParametersConfig:
    """Parameter configuration for convert arcade format flow."""

    box: tuple[int, int, int] = (1, 1, 1)

    frame_spec: tuple[int, int, int] = (0, 1153, 48)

    chunk_size: int = 500

    formats: list[str] = field(default_factory=lambda: FORMATS)

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])

    binary: bool = False

    separate: bool = False

    ds: float = 1.0

    dt: float = 1.0

    phase_colors: dict = field(default_factory=lambda: PHASE_COLORS)

    resolution: Optional[int] = None

    features: list[str] = field(default_factory=lambda: FEATURES)


@dataclass
class ContextConfig:
    """Context configuration for convert arcade format flow."""

    working_location: str


@dataclass
class SeriesConfig:
    """Series configuration for convert arcade format flow."""

    name: str

    seeds: list[int]

    conditions: list[dict]


@flow(name="convert-arcade-format")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    """Main convert arcade format flow."""

    if "colorizer" in parameters.formats:
        run_flow_convert_to_colorizer(context, series, parameters)

    if "image" in parameters.formats:
        run_flow_convert_to_images(context, series, parameters)

    if "mesh" in parameters.formats:
        run_flow_convert_to_meshes(context, series, parameters)

    if "simularium" in parameters.formats:
        run_flow_convert_to_simularium(context, series, parameters)


@flow(name="convert-arcade-format_convert-to-colorizer")
def run_flow_convert_to_colorizer(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    data_key = make_key(series.name, "data", "data.LOCATIONS")
    converted_key = make_key(series.name, "converted", "converted.COLORIZER")
    keys = [condition["key"] for condition in series.conditions]

    for key in keys:
        for seed in series.seeds:
            series_key = f"{series.name}_{key}_{seed:04d}"

            tar_key = make_key(data_key, f"{series_key}.LOCATIONS.tar.xz")
            tar = load_tar(context.working_location, tar_key)

            chunks = convert_to_images(
                series_key,
                tar,
                parameters.frame_spec,
                parameters.regions,
                parameters.box,
                parameters.chunk_size,
                binary=False,
                separate=True,
                flatten=True,
            )

            for frame_index, (_, _, chunk, _) in enumerate(chunks):
                image_key = make_key(converted_key, series_key, f"frame_{frame_index}.png")
                save_image(context.working_location, image_key, chunk)

            results_key = make_key(series.name, "results", f"{series_key}.csv")
            results = load_dataframe(context.working_location, results_key)

            colorizer = convert_to_colorizer(
                results,
                parameters.features,
                parameters.frame_spec,
                parameters.ds,
                parameters.dt,
                parameters.regions,
            )

            manifest_key = make_key(converted_key, series_key, "manifest.json")
            save_json(context.working_location, manifest_key, colorizer["manifest"])

            outliers_key = make_key(converted_key, series_key, "outliers.json")
            save_json(context.working_location, outliers_key, colorizer["outliers"])

            tracks_key = make_key(converted_key, series_key, "tracks.json")
            save_json(context.working_location, tracks_key, colorizer["tracks"])

            times_key = make_key(converted_key, series_key, "times.json")
            save_json(context.working_location, times_key, colorizer["times"])

            for feature_index, feature in enumerate(parameters.features):
                feature_key = make_key(converted_key, series_key, f"feature_{feature_index}.json")
                save_json(context.working_location, feature_key, colorizer[feature])


@flow(name="convert-arcade-format_convert-to-images")
def run_flow_convert_to_images(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    data_key = make_key(series.name, "data", "data.LOCATIONS")
    converted_key = make_key(series.name, "converted", "converted.IMAGE")
    keys = [condition["key"] for condition in series.conditions]

    for key in keys:
        for seed in series.seeds:
            series_key = f"{series.name}_{key}_{seed:04d}"
            tar_key = make_key(data_key, f"{series_key}.LOCATIONS.tar.xz")
            tar = load_tar(context.working_location, tar_key)

            chunks = convert_to_images(
                series_key,
                tar,
                parameters.frame_spec,
                parameters.regions,
                parameters.box,
                parameters.chunk_size,
                binary=parameters.binary,
                separate=parameters.separate,
                flatten=False,
            )

            for i, j, chunk, frame in chunks:
                chunk_key = f"{i:02d}_{j:02d}.IMAGE.ome.tiff"

                if frame is None:
                    image_key = make_key(converted_key, f"{series_key}_{chunk_key}")
                else:
                    image_key = make_key(converted_key, f"{series_key}_{frame:06d}_{chunk_key}")

                save_image(context.working_location, image_key, chunk)


@flow(name="convert-arcade-format_convert-to-meshes")
def run_flow_convert_to_meshes(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    data_key = make_key(series.name, "data", "data.LOCATIONS")
    converted_key = make_key(series.name, "converted", "converted.MESH")
    keys = [condition["key"] for condition in series.conditions]

    for key in keys:
        for seed in series.seeds:
            series_key = f"{series.name}_{key}_{seed:04d}"
            tar_key = make_key(data_key, f"{series_key}.LOCATIONS.tar.xz")
            tar = load_tar(context.working_location, tar_key)

            meshes = convert_to_meshes(series_key, tar, parameters.frame_spec, parameters.regions)

            for frame, cell_id, region, mesh in meshes:
                region_key = f"_{region}" if region != "DEFAULT" else ""
                mesh_key = make_key(
                    converted_key, f"{series_key}_{frame:06d}_{cell_id:02d}{region_key}.MESH.obj"
                )
                save_text(context.working_location, mesh_key, mesh)


@flow(name="convert-arcade-format_convert-to-simularium")
def run_flow_convert_to_simularium(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    cells_data_key = make_key(series.name, "data", "data.CELLS")
    locs_data_key = make_key(series.name, "data", "data.LOCATIONS")
    converted_key = make_key(series.name, "converted", "converted.SIMULARIUM")
    keys = [condition["key"] for condition in series.conditions]

    for key in keys:
        for seed in series.seeds:
            series_key = f"{series.name}_{key}_{seed:04d}"
            cells_tar_key = make_key(cells_data_key, f"{series_key}.CELLS.tar.xz")
            cells_tar = load_tar(context.working_location, cells_tar_key)
            locs_tar_key = make_key(locs_data_key, f"{series_key}.LOCATIONS.tar.xz")
            locs_tar = load_tar(context.working_location, locs_tar_key)

            simularium = convert_to_simularium(
                series_key,
                cells_tar,
                locs_tar,
                parameters.frame_spec,
                parameters.box,
                parameters.ds,
                parameters.dt,
                parameters.phase_colors,
                parameters.resolution,
            )
            simularium_key = make_key(converted_key, f"{series_key}.simularium")
            save_text(context.working_location, simularium_key, simularium)
