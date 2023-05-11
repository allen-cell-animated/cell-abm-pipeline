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
    plot_feature_average,
    plot_feature_distribution,
    plot_feature_individual,
    plot_feature_locations,
    plot_feature_merge,
    plot_phase_durations,
    plot_phase_fractions,
    plot_phase_locations,
    plot_population_locations,
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

BIN_SIZES: dict[str, float] = {
    "volume": 100,
    "height": 1,
    "volume.NUCLEUS": 50,
    "height.NUCLEUS": 1,
}


@dataclass
class ParametersConfig:
    reference: Optional[str] = None

    region: Optional[str] = None

    ds: float = 1.0

    dt: float = 1.0

    tick: int = 0

    chunk: int = 5

    plots: list[str] = field(default_factory=lambda: PLOTS)

    phases: list[str] = field(default_factory=lambda: CELL_PHASES)

    phase_colors: dict[str, str] = field(default_factory=lambda: PHASE_COLORS)

    bin_sizes: dict[str, float] = field(default_factory=lambda: BIN_SIZES)

    ids: Optional[list[int]] = field(default_factory=lambda: [1])


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

    height_feature = f"height.{parameters.region}" if parameters.region else "height"
    volume_feature = f"volume.{parameters.region}" if parameters.region else "volume"

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
            plot_feature_average(keys, height_feature, all_results, reference_data),
        )

    if "height_distribution" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_height_distribution{region_key}.BASIC.png"),
            plot_feature_distribution(
                keys, height_feature, all_results, parameters.bin_sizes[feature], reference_data
            ),
        )

    if "height_individual" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_height_individual{region_key}.BASIC.png"),
            plot_feature_individual(keys, height_feature, all_results, parameters.ids),
        )

    if "height_merge" in parameters.plots and reference_data is not None:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_height_merge.BASIC.png"),
            plot_feature_merge(
                keys,
                "height",
                all_results,
                parameters.bin_sizes,
                reference_data,
                parameters.region,
                parameters.ordered,
            ),
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
            plot_volume_average(keys, volume_feature, all_results, reference_data),
        )

    if "volume_distribution" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_volume_distribution{region_key}.BASIC.png"),
            plot_feature_distribution(
                keys, volume_feature, all_results, parameters.bin_sizes[feature], reference_data
            ),
        )

    if "volume_individual" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_volume_individual{region_key}.BASIC.png"),
            plot_feature_individual(keys, volume_feature, all_results, parameters.ids),
        )

    if "volume_merge" in parameters.plots and reference_data is not None:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_volume_merge.BASIC.png"),
            plot_feature_merge(
                keys,
                "volume",
                all_results,
                parameters.bin_sizes,
                reference_data,
                parameters.region,
                parameters.ordered,
            ),
        )


@flow(name="plot-basic-metrics_plot-spatial")
def run_flow_plot_spatial(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    plot_key = make_key(series.name, "plots", "plots.BASIC")
    region_key = f"_{parameters.region}" if parameters.region is not None else ""

    height_feature = f"height.{parameters.region}" if parameters.region else "height"
    volume_feature = f"volume.{parameters.region}" if parameters.region else "volume"

    for index in range(0, len(series.seeds), parameters.chunk):
        start = index
        end = index + parameters.chunk

        subset_key = f"T{parameters.tick:06d}_{series.seeds[start]:04d}_{series.seeds[end - 1]:04d}"
        subset_region_key = f"{subset_key}{region_key}"

        keys = [
            (condition["key"], seed)
            for condition in series.conditions
            for seed in series.seeds[start:end]
        ]

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
                make_key(plot_key, f"{series.name}_height_locations_{subset_region_key}.BASIC.png"),
                plot_feature_locations(
                    keys, height_feature, all_results, parameters.tick, reference_data
                ),
            )

        if "phase_locations" in parameters.plots and parameters.region is None:
            save_figure(
                context.working_location,
                make_key(plot_key, f"{series.name}_phase_locations_{subset_key}.BASIC.png"),
                plot_phase_locations(keys, all_results, parameters.tick, parameters.phase_colors),
            )

        if "population_locations" in parameters.plots and parameters.region is None:
            save_figure(
                context.working_location,
                make_key(plot_key, f"{series.name}_population_locations_{subset_key}.BASIC.png"),
                plot_population_locations(keys, all_results, parameters.tick),
            )

        if "volume_locations" in parameters.plots:
            save_figure(
                context.working_location,
                make_key(plot_key, f"{series.name}_volume_locations_{subset_region_key}.BASIC.png"),
                plot_feature_locations(
                    keys, volume_feature, all_results, parameters.tick, reference_data
                ),
            )
