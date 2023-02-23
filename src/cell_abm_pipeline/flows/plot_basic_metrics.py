from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from arcade_collection.output import convert_model_units
from io_collection.keys import make_key
from io_collection.load import load_dataframe
from io_collection.save import save_figure
from prefect import flow

from cell_abm_pipeline.tasks.basic import (
    plot_counts_total,
    plot_height_average,
    plot_height_distribution,
    plot_height_individual,
    plot_height_locations,
    plot_height_merge,
    plot_phase_durations,
    plot_phase_fractions,
    plot_phase_locations,
    plot_population_locations,
    plot_volume_average,
    plot_volume_distribution,
    plot_volume_individual,
    plot_volume_locations,
    plot_volume_merge,
)

PLOTS_TEMPORAL = [
    "counts_total",
    "height_average",
    "height_distribution",
    "height_individual",
    "height_merge",
    "phase_durations",
    "phase_fractions",
    "volume_average",
    "volume_distribution",
    "volume_individual",
    "volume_merge",
]

PLOTS_SPATIAL = [
    "height_locations",
    "phase_locations",
    "population_locations",
    "volume_locations",
]

PLOTS = PLOTS_TEMPORAL + PLOTS_SPATIAL

CELL_PHASES = [
    "PROLIFERATIVE_G1",
    "PROLIFERATIVE_S",
    "PROLIFERATIVE_G2",
    "PROLIFERATIVE_M",
    "APOPTOTIC_EARLY",
    "APOPTOTIC_LATE",
]

PHASE_COLORS: dict[str, str] = {
    "PROLIFERATIVE_G1": "#5F4690",
    "PROLIFERATIVE_S": "#38A6A5",
    "PROLIFERATIVE_G2": "#73AF48",
    "PROLIFERATIVE_M": "#CC503E",
    "APOPTOTIC_EARLY": "#E17C05",
    "APOPTOTIC_LATE": "#94346E",
}

BOUNDS: dict[str, tuple[int, int]] = {
    "volume": (0, 5500),
    "height": (0, 23),
    "volume.NUCLEUS": (0, 1600),
    "height.NUCLEUS": (0, 18),
}

SUBSET_THRESHOLDS: list[int] = [24, 48, 72]


@dataclass
class ParametersConfig:
    reference: Optional[str] = None

    region: Optional[str] = None

    ds: float = 1.0

    dt: float = 1.0

    tick: int = 0

    plots: list[str] = field(default_factory=lambda: PLOTS)

    phases: list[str] = field(default_factory=lambda: CELL_PHASES)

    phase_colors: dict[str, str] = field(default_factory=lambda: PHASE_COLORS)

    bounds: dict = field(default_factory=lambda: BOUNDS)

    thresholds: list[int] = field(default_factory=lambda: SUBSET_THRESHOLDS)


@dataclass
class ContextConfig:
    working_location: str


@dataclass
class SeriesConfig:
    name: str

    seeds: list[int]

    conditions: list[dict]


@flow(name="plot-basic-metrics")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    # Make plots for basic temporal metrics.
    if any(plot in parameters.plots for plot in PLOTS_TEMPORAL):
        run_flow_plot_temporal(context, series, parameters)

    # Make plots for basic spatial metrics.
    if any(plot in parameters.plots for plot in PLOTS_SPATIAL):
        run_flow_plot_spatial(context, series, parameters)


@flow(name="plot-basic-metrics_plot-temporal")
def run_flow_plot_temporal(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    plot_key = make_key(series.name, "plots", "plots.BASIC")
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
            convert_model_units(results, parameters.ds, parameters.dt, parameters.region)
            key_results.append(results)

        all_results[key] = pd.concat(key_results)

    reference_data = None
    if parameters.reference:
        reference_data = load_dataframe(context.working_location, parameters.reference)

    if "counts_total" in parameters.plots and parameters.region is None:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_counts_total.BASIC.png"),
            plot_counts_total(keys, all_results),
        )

    if "height_average" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_height_average{region_key}.BASIC.png"),
            plot_height_average(keys, all_results, reference_data, parameters.region),
        )

    if "height_distribution" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_height_distribution{region_key}.BASIC.png"),
            plot_height_distribution(
                keys,
                all_results,
                parameters.bounds,
                reference_data,
                parameters.region,
                parameters.thresholds,
            ),
        )

    if "height_individual" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_height_individual{region_key}.BASIC.png"),
            plot_height_individual(keys, all_results, parameters.region),
        )

    if "height_merge" in parameters.plots and reference_data is not None:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_height_merge.BASIC.png"),
            plot_height_merge(keys, all_results, reference_data, parameters.region),
        )

    if "phase_durations" in parameters.plots and parameters.region is None:
        for phase in parameters.phases:
            phase_plot = plot_phase_durations(keys, all_results, phase, parameters.phase_colors)

            if phase_plot is None:
                continue

            save_figure(
                context.working_location,
                make_key(plot_key, f"{series.name}_phase_durations_{phase}.BASIC.png"),
                phase_plot,
            )

    if "phase_fractions" in parameters.plots and parameters.region is None:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_phase_fractions.BASIC.png"),
            plot_phase_fractions(keys, all_results, parameters.phases, parameters.phase_colors),
        )

    if "volume_average" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_volume_average{region_key}.BASIC.png"),
            plot_volume_average(keys, all_results, reference_data, parameters.region),
        )

    if "volume_distribution" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_volume_distribution{region_key}.BASIC.png"),
            plot_volume_distribution(
                keys,
                all_results,
                parameters.bounds,
                reference_data,
                parameters.region,
                parameters.thresholds,
            ),
        )

    if "volume_individual" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_volume_individual{region_key}.BASIC.png"),
            plot_volume_individual(keys, all_results, parameters.region),
        )

    if "volume_merge" in parameters.plots and reference_data is not None:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_volume_merge.BASIC.png"),
            plot_volume_merge(keys, all_results, reference_data, parameters.region),
        )


@flow(name="plot-basic-metrics_plot-spatial")
def run_flow_plot_spatial(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    plot_key = make_key(series.name, "plots", "plots.BASIC")
    region_key = f"_{parameters.region}" if parameters.region is not None else ""
    tick_key = f"T{parameters.tick:06d}"
    keys = [(condition["key"], seed) for condition in series.conditions for seed in series.seeds]

    all_results = {}

    for key, seed in keys:
        series_key = f"{series.name}_{key}_{seed:04d}"
        results_key = make_key(series.name, "results", f"{series_key}.csv")
        results = load_dataframe(context.working_location, results_key)
        convert_model_units(results, parameters.ds, parameters.dt, parameters.region)
        all_results[(key, seed)] = results

    reference_data = None
    if parameters.reference:
        reference_data = load_dataframe(context.working_location, parameters.reference)

    if "height_locations" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_height_locations_{tick_key}{region_key}.BASIC.png"),
            plot_height_locations(
                keys, all_results, parameters.tick, reference_data, parameters.region
            ),
        )

    if "phase_locations" in parameters.plots and parameters.region is None:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_phase_locations_{tick_key}.BASIC.png"),
            plot_phase_locations(keys, all_results, parameters.tick, parameters.phase_colors),
        )

    if "population_locations" in parameters.plots and parameters.region is None:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_population_locations_{tick_key}.BASIC.png"),
            plot_population_locations(keys, all_results, parameters.tick),
        )

    if "volume_locations" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_volume_locations_{tick_key}{region_key}.BASIC.png"),
            plot_volume_locations(
                keys, all_results, parameters.tick, reference_data, parameters.region
            ),
        )
