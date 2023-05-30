from dataclasses import dataclass, field

from io_collection.keys import make_key
from io_collection.load import load_tar
from io_collection.save import save_text

from ..tasks.physicell import convert_physicell_to_simularium

from prefect import flow

FORMATS: list[str] = [
    "simularium",
]


@dataclass
class ParametersConfig:
    box_size: list[float]
    timestep: float
    formats: list[str] = field(default_factory=lambda: FORMATS)


@dataclass
class ContextConfig:
    working_location: str


@dataclass
class SeriesConfig:
    name: str

    seeds: list[int]

    conditions: list[dict]


@flow(name="convert-physicell-format")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    if "simularium" in parameters.formats:
        run_flow_convert_to_simularium(context, series, parameters)


@flow(name="convert-physicell-format_convert-to-simularium")
def run_flow_convert_to_simularium(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    data_key = make_key(series.name, "data")
    converted_key = make_key(series.name, "converted", "converted.SIMULARIUM")
    keys = [condition["key"] for condition in series.conditions]

    for key in keys:
        for seed in series.seeds:
            series_key = f"{series.name}_{key}_{seed:04d}"
            tar_key = make_key(data_key, f"{series_key}.CELLS.tar.xz")
            tar_file = load_tar(context.working_location, tar_key)
            
            json_str = convert_physicell_to_simularium(tar_file, parameters.box_size, parameters.timestep)

            simularium_key = make_key(converted_key, f"{series_key}.simularium")
            save_text(context.working_location, simularium_key, json_str)

