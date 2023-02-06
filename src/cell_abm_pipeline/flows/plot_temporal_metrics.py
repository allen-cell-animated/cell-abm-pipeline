from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from io_collection.keys import make_key
from io_collection.load import load_dataframe
from io_collection.save import save_figure
from prefect import flow

from cell_abm_pipeline.tasks import (
    convert_data_units,
    plot_counts_total,
    plot_height_average,
    plot_height_distribution,
    plot_height_individual,
    plot_phase_durations,
    plot_phase_fractions,
    plot_volume_average,
    plot_volume_distribution,
    plot_volume_individual,
)

PLOTS = [
    "counts_total",
    "height_average",
    "height_distribution",
    "height_individual",
    "phase_durations",
    "phase_fractions",
    "volume_average",
    "volume_distribution",
    "volume_individual",
]

CELL_PHASES = [
    "PROLIFERATIVE_G1",
    "PROLIFERATIVE_S",
    "PROLIFERATIVE_G2",
    "PROLIFERATIVE_M",
    "APOPTOTIC_EARLY",
    "APOPTOTIC_LATE",
]


@dataclass
class ParametersConfig:
    reference: Optional[str] = None

    region: Optional[str] = None

    ds: float = 1.0

    dt: float = 1.0

    plots: list[str] = field(default_factory=lambda: PLOTS)

    phases: list[str] = field(default_factory=lambda: CELL_PHASES)


@dataclass
class ContextConfig:
    working_location: str


@dataclass
class SeriesConfig:
    name: str

    seeds: list[int]

    conditions: list[dict]


@flow(name="plot-temporal-metrics")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    region_key = f"_{parameters.region}" if parameters.region is not None else ""
    keys = [condition["key"] for condition in series.conditions]

    all_results = {}

    for key in keys:
        key_results = []

        for seed in series.seeds:
            series_key = f"{series.name}_{key}_{seed:04d}"
            results_key = make_key(series.name, "results", f"{series_key}.csv")
            results = load_dataframe(context.working_location, results_key)
            results["SEED"] = seed
            convert_data_units(results, parameters.ds, parameters.dt, parameters.region)
            key_results.append(results)

        all_results[key] = pd.concat(key_results)

    reference_dataframe = None
    if parameters.reference:
        reference_dataframe = load_dataframe(context.working_location, parameters.reference)

    plot_key = make_key(series.name, "plots", "plots.BASIC")

    if "counts_total" in parameters.plots and parameters.region is None:
        save_figure(
            context.working_location,
            make_key(plot_key, "counts_total.png"),
            plot_counts_total(keys, all_results),
        )

    if "height_average" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"height_average{region_key}.png"),
            plot_height_average(keys, all_results, reference_dataframe, parameters.region),
        )

    if "height_individual" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"height_individual{region_key}.png"),
            plot_height_individual(keys, all_results, parameters.region),
        )

    if "height_distribution" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"height_distribution{region_key}.png"),
            plot_height_distribution(keys, all_results, reference_dataframe, parameters.region),
        )

    if "phase_durations" in parameters.plots and parameters.region is None:
        for phase in parameters.phases:
            phase_plot = plot_phase_durations(keys, all_results, phase)

            if phase_plot is None:
                continue

            save_figure(
                context.working_location,
                make_key(plot_key, f"phase_durations_{phase}.png"),
                phase_plot,
            )

    if "phase_fractions" in parameters.plots and parameters.region is None:
        save_figure(
            context.working_location,
            make_key(plot_key, "phase_fractions.png"),
            plot_phase_fractions(keys, all_results, parameters.phases),
        )

    if "volume_average" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"volume_average{region_key}.png"),
            plot_volume_average(keys, all_results, reference_dataframe, parameters.region),
        )

    if "volume_distribution" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"volume_distribution{region_key}.png"),
            plot_volume_distribution(keys, all_results, reference_dataframe, parameters.region),
        )

    if "volume_individual" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"volume_individual{region_key}.png"),
            plot_volume_individual(keys, all_results, parameters.region),
        )
