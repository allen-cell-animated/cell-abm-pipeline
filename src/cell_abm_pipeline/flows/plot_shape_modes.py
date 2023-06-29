"""
Workflow for plotting shape modes.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from abm_shape_collection import compile_shape_modes, extract_shape_modes, merge_shape_modes
from arcade_collection.output import extract_tick_json
from io_collection.keys import make_key
from io_collection.load import load_dataframe, load_pickle, load_tar
from io_collection.save import save_figure, save_text
from prefect import flow

from cell_abm_pipeline.flows.analyze_shape_modes import PCA_COMPONENTS
from cell_abm_pipeline.flows.calculate_coefficients import COEFFICIENT_ORDER
from cell_abm_pipeline.flows.plot_basic_metrics import BIN_SIZES
from cell_abm_pipeline.tasks.basic import plot_feature_distribution, plot_feature_merge
from cell_abm_pipeline.tasks.pca import (
    plot_correlation_all_features,
    plot_correlation_pca_features,
    plot_correlation_region_features,
    plot_variance_explained,
)
from cell_abm_pipeline.tasks.shapes import plot_sample_shapes
from cell_abm_pipeline.tasks.stats import (
    plot_ks_all_ticks,
    plot_ks_by_feature,
    plot_ks_by_key,
    plot_ks_by_sample,
)

PLOTS_PCA = [
    "feature_compare",
    "feature_merge",
    "feature_correlation",
    "transform_compare",
    "transform_merge",
    "variance_explained",
]

PLOTS_STATS = [
    "ks_stats",
]

PLOTS_SHAPES = [
    "sample_shapes",
    "shape_modes_compile",
    "shape_modes_merge",
]

PLOTS = PLOTS_PCA + PLOTS_STATS + PLOTS_SHAPES

REGION_COLORS = {"DEFAULT": "#F200FF", "NUCLEUS": "#3AADA7"}

MODE_VIEWS = ["top", "side_1", "side_2"]

ALL_BIN_SIZES: dict[str, float] = {
    "volume": BIN_SIZES["volume"],
    "height": BIN_SIZES["height"],
    "volume.NUCLEUS": BIN_SIZES["volume.NUCLEUS"],
    "height.NUCLEUS": BIN_SIZES["height.NUCLEUS"],
    "PC1": 1,
    "PC2": 1,
    "PC3": 1,
    "PC4": 1,
    "PC5": 1,
    "PC6": 1,
    "PC7": 1,
    "PC8": 1,
}


@dataclass
class ParametersConfig:
    """Parameter configuration for plot shape modes flow."""

    reference: Optional[dict] = None

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])

    plots: list[str] = field(default_factory=lambda: PLOTS)

    components: int = PCA_COMPONENTS
    """Number of principal components (i.e. shape modes)."""

    order: int = COEFFICIENT_ORDER
    """Order of the spherical harmonics coefficient parametrization."""

    bin_sizes: dict[str, float] = field(default_factory=lambda: ALL_BIN_SIZES)

    delta: float = 1.0

    box: tuple[int, int] = (150, 100)

    scale: float = 0.5

    colors: dict = field(default_factory=lambda: REGION_COLORS)

    views: list = field(default_factory=lambda: MODE_VIEWS)

    ordered: bool = True

    sample_seed: int = 0

    sample_tick: int = 0

    sample_size: int = 100

    sample_box: tuple[int, int, int] = (22, 60, 60)

    random_seed: int = 0


@dataclass
class ContextConfig:
    """Context configuration for plot shape modes flow."""

    working_location: str


@dataclass
class SeriesConfig:
    """Series configuration for plot shape modes flow."""

    name: str

    conditions: list[dict]


@flow(name="plot-shape-modes")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    # Make plots for PCA results including variance explained and comparisons of
    # distributions in PC space using the given reference model.
    if any(plot in parameters.plots for plot in PLOTS_PCA):
        run_flow_plot_pca(context, series, parameters)

    # Make plots for statistical comparisons of shape distributions.
    if any(plot in parameters.plots for plot in PLOTS_STATS):
        run_flow_plot_stats(context, series, parameters)

    # Make plots showing shape modes.
    if any(plot in parameters.plots for plot in PLOTS_SHAPES):
        run_flow_plot_shapes(context, series, parameters)


@flow(name="plot-shape-modes_plot-pca")
def run_flow_plot_pca(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    analysis_key = make_key(series.name, "analysis", "analysis.PCA")
    plot_key = make_key(series.name, "plots", "plots.PCA")
    region_key = ":".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

    all_models = {}
    all_data = {}

    for key in keys:
        model_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.PCA.pkl")
        model = load_pickle(context.working_location, model_key)
        all_models[key] = model

        data_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.PCA.csv")
        data = load_dataframe(context.working_location, data_key)
        all_data[key] = data

    ref_model = None
    ref_data = None

    if parameters.reference is not None:
        ref_model = load_pickle(context.working_location, parameters.reference["model"])
        ref_data = load_dataframe(context.working_location, parameters.reference["data"])

        columns = ref_data.filter(like="shcoeffs").columns
        ref_transform = ref_model.transform(ref_data[columns].values)

        for component in range(parameters.components):
            ref_data[f"PC{component + 1}"] = ref_transform[:, component]

        for key in keys:
            key_transform = ref_model.transform(all_data[key][columns].values)

            for component in range(parameters.components):
                all_data[key][f"PC{component + 1}"] = key_transform[:, component]

    if "feature_compare" in parameters.plots:
        for region in parameters.regions:
            for feature in ["volume", "height"]:
                feature_name = f"{feature}.{region}" if region != "DEFAULT" else feature
                save_figure(
                    context.working_location,
                    make_key(plot_key, f"{series.name}_feature_compare_{feature}_{region}.PCA.png"),
                    plot_feature_distribution(
                        keys, feature_name, all_data, parameters.bin_sizes[feature_name], ref_data
                    ),
                )

    if "feature_correlation" in parameters.plots:
        for key in keys:
            save_figure(
                context.working_location,
                make_key(
                    plot_key, f"{series.name}_correlation_all_features_{key}_{region_key}.PCA.png"
                ),
                plot_correlation_all_features(all_models[key], all_data[key], parameters.regions),
            )

            save_figure(
                context.working_location,
                make_key(
                    plot_key, f"{series.name}_correlation_pca_features_{key}_{region_key}.PCA.png"
                ),
                plot_correlation_pca_features(all_models[key], all_data[key]),
            )

            save_figure(
                context.working_location,
                make_key(
                    plot_key,
                    f"{series.name}_correlation_region_features_{key}_{region_key}.PCA.png",
                ),
                plot_correlation_region_features(all_data[key], parameters.regions),
            )

    if "feature_merge" in parameters.plots:
        for feature in ["volume", "height"]:
            save_figure(
                context.working_location,
                make_key(plot_key, f"{series.name}_feature_merge_{feature}.PCA.png"),
                plot_feature_merge(
                    keys,
                    feature,
                    all_data,
                    parameters.bin_sizes,
                    [
                        None if region == "DEFAULT" else f".{region}"
                        for region in parameters.regions
                    ],
                    ref_data,
                    parameters.ordered,
                ),
            )

    if "transform_compare" in parameters.plots and ref_model is not None:
        for component in range(parameters.components):
            pci = f"PC{component + 1}"
            save_figure(
                context.working_location,
                make_key(plot_key, f"{series.name}_transform_compare_{pci}_{region_key}.PCA.png"),
                plot_feature_distribution(
                    keys, pci, all_data, parameters.bin_sizes[pci], ref_data, symmetric=True
                ),
            )

    if "transform_merge" in parameters.plots and ref_model is not None:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_transform_merge_{region_key}.PCA.png"),
            plot_feature_merge(
                keys,
                "",
                all_data,
                parameters.bin_sizes,
                [f"PC{i + 1}" for i in range(8)],
                ref_data,
                parameters.ordered,
                symmetric=True,
            ),
        )

    if "variance_explained" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_variance_explained_{region_key}.PCA.png"),
            plot_variance_explained(keys, all_models, ref_model),
        )


@flow(name="plot-shape-modes_plot-stats")
def run_flow_plot_stats(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    analysis_key = make_key(series.name, "analysis", "analysis.STATS")
    plot_key = make_key(series.name, "plots", "plots.STATS")
    region_key = ":".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

    all_stats = []

    for key in keys:
        stats_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.STATS.csv")
        stats = load_dataframe(context.working_location, stats_key)
        stats["KEY"] = key
        all_stats.append(stats)

    all_stats = pd.concat(all_stats)

    ref_stats = None

    if parameters.reference is not None:
        ref_stats = load_dataframe(context.working_location, parameters.reference["stats"])

    if "ks_stats" in parameters.plots:
        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_ks_stats_all_ticks.STATS.png"),
            plot_ks_all_ticks(keys, all_stats, parameters.ordered),
        )

        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_ks_stats_by_key.STATS.png"),
            plot_ks_by_key(keys, all_stats),
        )

        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_ks_stats_by_feature.STATS.png"),
            plot_ks_by_feature(keys, all_stats, ref_stats, parameters.ordered),
        )

        save_figure(
            context.working_location,
            make_key(plot_key, f"{series.name}_ks_stats_by_sample.STATS.png"),
            plot_ks_by_sample(keys, all_stats, parameters.ordered),
        )


@flow(name="plot-shape-modes_plot-shapes")
def run_flow_plot_shapes(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    data_key = make_key(series.name, "data", "data.LOCATIONS")
    analysis_key = make_key(series.name, "analysis", "analysis.PCA")
    plot_key = make_key(series.name, "plots", "plots.SHAPES")
    region_key = ":".join(sorted(parameters.regions))
    views_key = ":".join(sorted(parameters.views))
    keys = [condition["key"] for condition in series.conditions]

    for key in keys:
        if "shape_modes_compile" in parameters.plots or "shape_modes_merge" in parameters.plots:
            pca_data_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.PCA.csv")
            pca_data = load_dataframe(context.working_location, pca_data_key)

            pca_model_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.PCA.pkl")
            pca_model = load_pickle(context.working_location, pca_model_key)

            shape_modes = extract_shape_modes(
                pca_model,
                pca_data,
                parameters.components,
                parameters.regions,
                parameters.order,
                parameters.delta,
            )

        if "shape_modes_compile" in parameters.plots:
            shape_modes_compile = compile_shape_modes(
                shape_modes,
                parameters.views,
                parameters.regions,
                pca_model.explained_variance_ratio_,
                parameters.colors,
                parameters.box,
                parameters.scale,
            )
            shape_modes_compile_key = make_key(
                plot_key,
                f"{series.name}_{key}_shape_modes_compile_{region_key}_{views_key}.SHAPES.svg",
            )
            save_text(context.working_location, shape_modes_compile_key, shape_modes_compile)

        if "shape_modes_merge" in parameters.plots:
            shape_modes_merge = merge_shape_modes(
                shape_modes,
                parameters.views,
                parameters.regions,
                pca_model.explained_variance_ratio_,
                parameters.colors,
                parameters.box,
                parameters.scale,
            )
            shape_modes_merge_key = make_key(
                plot_key,
                f"{series.name}_{key}_shape_modes_merge_{region_key}_{views_key}.SHAPES.svg",
            )
            save_text(context.working_location, shape_modes_merge_key, shape_modes_merge)

        if "sample_shapes" in parameters.plots:
            tick_key = f"{series.name}_{key}_{parameters.sample_seed:04d}"
            locations_key = make_key(data_key, f"{tick_key}.LOCATIONS.tar.xz")
            locations_tar = load_tar(context.working_location, locations_key)
            locations = extract_tick_json(
                locations_tar, tick_key, parameters.sample_tick, "LOCATIONS"
            )

            for region in parameters.regions:
                subset_key = f"T{parameters.sample_tick:06d}_{parameters.sample_seed:04d}_{region}"
                save_figure(
                    context.working_location,
                    make_key(
                        plot_key, f"{series.name}_{key}_sample_shapes_{subset_key}.SHAPES.png"
                    ),
                    plot_sample_shapes(
                        locations,
                        parameters.sample_box,
                        region,
                        parameters.sample_size,
                        parameters.random_seed,
                    ),
                )
