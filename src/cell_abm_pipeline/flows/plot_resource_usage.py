"""
Workflow for plotting resource usage.
"""

from dataclasses import dataclass, field

from io_collection.keys import make_key
from io_collection.load import load_dataframe
from io_collection.save import save_figure
from prefect import flow

from cell_abm_pipeline.flows.analyze_resource_usage import STORAGE_GROUPS
from cell_abm_pipeline.tasks.resources import plot_object_storage, plot_wall_clock

PLOTS = [
    "object_storage",
    "wall_clock",
]


@dataclass
class ParametersConfig:
    groups: list[str] = field(default_factory=lambda: STORAGE_GROUPS)

    plots: list[str] = field(default_factory=lambda: PLOTS)


@dataclass
class ContextConfig:
    working_location: str


@dataclass
class SeriesConfig:
    name: str

    conditions: list[dict]


@flow(name="plot-resource-usage")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    analysis_key = make_key(series.name, "analysis", "analysis.RESOURCES")
    plot_key = make_key(series.name, "plots", "plots.RESOURCES")
    keys = [condition["key"] for condition in series.conditions]

    if "object_storage" in parameters.plots:
        storage_key = make_key(analysis_key, f"{series.name}_object_storage.RESOURCES.csv")
        storage = load_dataframe(context.working_location, storage_key)

        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_object_storage.RESOURCES.png"),
            plot_object_storage(keys, storage, parameters.groups),
        )

    if "wall_clock" in parameters.plots:
        clock_key = make_key(analysis_key, f"{series.name}_wall_clock.RESOURCES.csv")
        clock = load_dataframe(context.working_location, clock_key)

        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_wall_clock.RESOURCES.png"),
            plot_wall_clock(keys, clock),
        )
