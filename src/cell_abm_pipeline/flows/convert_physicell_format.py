"""
Workflow for converting PhysiCell simulations to other formats.
"""

from dataclasses import dataclass, field

from io_collection.keys import make_key
from io_collection.load import load_tar
from io_collection.save import save_text
from prefect import flow

from cell_abm_pipeline.tasks.physicell import convert_physicell_to_simularium

FORMATS: list[str] = [
    "simularium",
]

SUBSTRATE_COLOR = "#d0c5c8"

CELL_COLORS = [
    "#fee34d",
    "#f7b232",
    "#bf5736",
    "#94a7fc",
    "#ce8ec9",
    "#58606c",
    "#0ba345",
    "#9267cb",
    "#81dbe6",
    "#bd7800",
    "#bbbb99",
    "#5b79f0",
    "#89a500",
    "#da8692",
    "#418463",
    "#9f516c",
    "#00aabf",
]


@dataclass
class ParametersConfig:
    """Parameter configuration for convert physicell format flow."""

    box_size: tuple[float, float, float] = (1.0, 1.0, 1.0)

    timestep: float = 1.0

    formats: list[str] = field(default_factory=lambda: FORMATS)

    substrate_color: str = SUBSTRATE_COLOR

    cell_colors: list[str] = field(default_factory=lambda: CELL_COLORS)

    frame_spec: tuple[int, int, int] = (0, 1, 1)


@dataclass
class ContextConfig:
    """Context configuration for convert physicell format flow."""

    working_location: str


@dataclass
class SeriesConfig:
    """Series configuration for convert physicell format flow."""

    name: str

    seeds: list[int]

    conditions: list[dict]


@flow(name="convert-physicell-format")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    """Main convert physicell format flow."""

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
            tar_key = make_key(data_key, f"{series_key}.tar.xz")
            tar_file = load_tar(context.working_location, tar_key)

            simularium = convert_physicell_to_simularium(
                tar_file,
                parameters.box_size,
                parameters.timestep,
                parameters.frame_spec,
                parameters.substrate_color,
                parameters.cell_colors,
            )
            simularium_key = make_key(converted_key, f"{series_key}.simularium")
            save_text(context.working_location, simularium_key, simularium)
