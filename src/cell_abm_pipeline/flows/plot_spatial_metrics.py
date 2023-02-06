from dataclasses import dataclass, field
from typing import Optional

from io_collection.keys import make_key
from io_collection.load import load_dataframe
from io_collection.save import save_figure
from prefect import flow

from cell_abm_pipeline.tasks import (
    convert_data_units,
    plot_height_locations,
    plot_phase_locations,
    plot_population_locations,
    plot_volume_locations,
)

PLOTS = [
    "height_locations",
    "phase_locations",
    "population_locations",
    "volume_locations",
]


@dataclass
class ParametersConfig:
    reference: Optional[str] = None

    region: Optional[str] = None

    ds: float = 1.0

    dt: float = 1.0

    tick: int = 0

    plots: list[str] = field(default_factory=lambda: PLOTS)


@dataclass
class ContextConfig:
    working_location: str


@dataclass
class SeriesConfig:
    name: str

    seeds: list[int]

    conditions: list[dict]


@flow(name="plot-spatial-metrics")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    region_key = f"_{parameters.region}" if parameters.region is not None else ""
    keys = [(condition["key"], seed) for condition in series.conditions for seed in series.seeds]

    all_results = {}

    for key, seed in keys:
        series_key = f"{series.name}_{key}_{seed:04d}"
        results_key = make_key(series.name, "results", f"{series_key}.csv")
        results = load_dataframe(context.working_location, results_key)
        convert_data_units(results, parameters.ds, parameters.dt, parameters.region)
        all_results[(key, seed)] = results

    reference_dataframe = None
    if parameters.reference:
        reference_dataframe = load_dataframe(context.working_location, parameters.reference)

    plot_key = make_key(series.name, "plots", "plots.BASIC")

    if "height_locations" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"height_locations{region_key}_T{parameters.tick:06d}"),
            plot_height_locations(
                keys, all_results, parameters.tick, reference_dataframe, parameters.region
            ),
        )

    if "phase_locations" in parameters.plots and parameters.region is None:
        save_figure(
            context.working_location,
            make_key(plot_key, f"phase_locations_T{parameters.tick:06d}"),
            plot_phase_locations(keys, all_results, parameters.tick),
        )

    if "population_locations" in parameters.plots and parameters.region is None:
        save_figure(
            context.working_location,
            make_key(plot_key, f"population_locations_T{parameters.tick:06d}"),
            plot_population_locations(keys, all_results, parameters.tick),
        )

    if "volume_locations" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"volume_locations{region_key}_T{parameters.tick:06d}"),
            plot_volume_locations(
                keys, all_results, parameters.tick, reference_dataframe, parameters.region
            ),
        )
