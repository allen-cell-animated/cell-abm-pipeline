"""
Workflow for converting PhysiCell simulations to other formats.

Working location structure:

.. code-block:: bash

    (name)
    ├── converted
    │   └── converted.SIMULARIUM
    │       └── (name)_(key)_(seed).simularium
    └── data
        └── (name)_(key)_(seed).tar.xz

Formatted data are saved to **converted**.
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
    """Parameter configuration for convert PhysiCell format flow."""

    box_size: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Size of bounding box."""

    timestep: float = 1.0
    """Simulation timestep."""

    formats: list[str] = field(default_factory=lambda: FORMATS)
    """List of convert formats."""

    substrate_color: str = SUBSTRATE_COLOR
    """Color for substrate."""

    cell_colors: list[str] = field(default_factory=lambda: CELL_COLORS)
    """Colors for individual cells."""

    frame_spec: tuple[int, int, int] = (0, 1, 1)
    """Specification for simulation ticks to use for converting to simularium."""


@dataclass
class ContextConfig:
    """Context configuration for convert PhysiCell format flow."""

    working_location: str
    """Location for input and output files (local path or S3 bucket)."""


@dataclass
class SeriesConfig:
    """Series configuration for convert PhysiCell format flow."""

    name: str
    """Name of the simulation series."""

    seeds: list[int]
    """List of series random seeds."""

    conditions: list[dict]
    """List of series condition dictionaries (must include unique condition "key")."""


@flow(name="convert-physicell-format")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    """Main convert PhysiCell format flow."""

    if "simularium" in parameters.formats:
        run_flow_convert_to_simularium(context, series, parameters)


@flow(name="convert-physicell-format_convert-to-simularium")
def run_flow_convert_to_simularium(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    """Convert PhysiCell format subflow for Simularium."""

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
