from dataclasses import dataclass, field

from arcade_collection.output import convert_to_images, convert_to_meshes
from io_collection.keys import make_key
from io_collection.load import load_tar
from io_collection.save import save_image, save_text
from prefect import flow

FORMATS = [
    "images",
    "meshes",
]


@dataclass
class ParametersConfig:
    box: tuple[int, int, int] = (1, 1, 1)

    frame_spec: tuple[int, int, int] = (0, 1153, 48)

    chunk_size: int = 500

    formats: list[str] = field(default_factory=lambda: FORMATS)

    regions: list[str] = field(default_factory=lambda: ["DEFAULT", "NUCLEUS"])

    binary: bool = False


@dataclass
class ContextConfig:
    working_location: str


@dataclass
class SeriesConfig:
    name: str

    seeds: list[int]

    conditions: list[dict]


@flow(name="convert-arcade-format")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    if "images" in parameters.formats:
        run_flow_convert_to_images(context, series, parameters)

    if "meshes" in parameters.formats:
        run_flow_convert_to_meshes(context, series, parameters)


@flow(name="convert-arcade-format_convert_to_images")
def run_flow_convert_to_images(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    converted_key = make_key(series.name, "converted", "converted.IMAGE")
    keys = [condition["key"] for condition in series.conditions]

    for key in keys:
        for seed in series.seeds:
            series_key = f"{series.name}_{key}_{seed:04d}"
            data_key = make_key(
                series.name, "data", "data.LOCATIONS", f"{series_key}.LOCATIONS.tar.xz"
            )
            data = load_tar(context.working_location, data_key)

            chunks = convert_to_images(
                series_key,
                data,
                parameters.frame_spec,
                parameters.regions,
                parameters.box,
                parameters.chunk_size,
                parameters.binary,
            )

            for i, j, chunk in chunks:
                image_key = make_key(converted_key, f"{series_key}_{i:02d}_{j:02d}.IMAGE.ome.tiff")
                save_image(context.working_location, image_key, chunk)


@flow(name="convert-arcade-format_convert_to_meshes")
def run_flow_convert_to_meshes(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    converted_key = make_key(series.name, "converted", "converted.MESH")
    keys = [condition["key"] for condition in series.conditions]

    for key in keys:
        for seed in series.seeds:
            series_key = f"{series.name}_{key}_{seed:04d}"
            data_key = make_key(
                series.name, "data", "data.LOCATIONS", f"{series_key}.LOCATIONS.tar.xz"
            )
            data = load_tar(context.working_location, data_key)

            meshes = convert_to_meshes(series_key, data, parameters.frame_spec, parameters.regions)

            for frame, cell_id, region, mesh in meshes:
                region_key = f"_{region}" if region != "DEFAULT" else ""
                mesh_key = make_key(
                    converted_key, f"{series_key}_{frame:06d}_{cell_id:02d}{region_key}.MESH.obj"
                )
                save_text(context.working_location, mesh_key, mesh)
