"""
Workflow for formatting figure panel data.

Working location structure:

.. code-block:: bash

    (name)
    ├── analysis
    │   ├── analysis.PCA
    │   │   ├── (name)_(key)_(regions).PCA.csv
    │   │   ├── (name)_(key)_(regions).PCA.pkl
    │   │   ├── (name)_(key)_(regions).PCA.csv
    │   │   ├── (name)_(key)_(regions).PCA.pkl
    │   │   ├── ...
    │   │   ├── ...
    │   │   ├── (name)_(key)_(regions).PCA.csv
    │   │   └── (name)_(key)_(regions).PCA.pkl
    │   └── analysis.STATS
    │       ├── HAMILTONIAN_TERMS_FOV_VSAxxxx_DEFAULT:NUCLEUS.STATS (copy).csv
    │       └── HAMILTONIAN_TERMS_FOV_VSAxxxx_DEFAULT:NUCLEUS.STATS.csv
    ├── data
    │   └── data.LOCATIONS
    │       ├── (name)_(key)_(seed).LOCATIONS.tar.xz
    │       ├── (name)_(key)_(seed).LOCATIONS.tar.xz
    │       ├── ...
    │       └── (name)_(key)_(seed).LOCATIONS.tar.xz
    ├── panels
    │   ├── (name).feature_bins.csv
    │   ├── (name).feature_correlations.csv
    │   ├── (name).feature_distributions.csv
    │   ├── (name).mode_correlations.csv
    │   ├── (name).population_counts.csv
    │   ├── (name).population_stats.csv
    │   ├── (name).shape_average.json
    │   ├── (name).shape_errors.csv
    │   ├── (name).shape_modes.json
    │   └── (name).shape_samples.json
    │   └── (name).variance_explained.csv
    └── results
        ├── (name)_(key)_(seed).csv
        ├── (name)_(key)_(seed).csv
        ├── ...
        └── (name)_(key)_(seed).csv

Different panels use inputs from the **results**, **data/data.LOCATION**,
**analysis/analysis.PCA**, and **analysis/analysis.STATS** directories.
Formatted panel data is saved to the **panels** directory.
"""

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd
from abm_shape_collection import (
    construct_mesh_from_array,
    construct_mesh_from_coeffs,
    extract_mesh_projections,
    extract_mesh_wireframe,
    extract_shape_modes,
    make_voxels_array,
)
from arcade_collection.output import extract_feature_bins, extract_tick_json, get_location_voxels
from io_collection.keys import make_key
from io_collection.load import load_dataframe, load_pickle, load_tar
from io_collection.save import save_dataframe, save_json
from prefect import flow, get_run_logger
from scipy.spatial import KDTree
from scipy.stats import pearsonr

from cell_abm_pipeline.flows.analyze_shape_modes import PCA_COMPONENTS
from cell_abm_pipeline.flows.calculate_coefficients import COEFFICIENT_ORDER

PANELS: list[str] = [
    "feature_bins",
    "feature_correlations",
    "feature_distributions",
    "mode_correlations",
    "population_counts",
    "population_stats",
    "shape_average",
    "shape_errors",
    "shape_modes",
    "shape_samples",
    "variance_explained",
]

FEATURES: list[str] = [
    "volume",
    "height",
    "area",
    "axis_major_length",
    "axis_minor_length",
    "eccentricity",
    "perimeter",
]


@dataclass
class ParametersConfigFeatureBins:
    """Parameter configuration for format panel data feature bins subflow."""

    seeds: list[int] = field(default_factory=lambda: [0])
    """Simulation seeds to use for formatting feature bins."""

    tick: int = 0
    """Simulation tick to use for formatting feature bins."""

    scale: float = 1
    """Feature bin scaling."""


@dataclass
class ParametersConfigFeatureCorrelations:
    """Parameter configuration for format panel data feature correlations subflow."""

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""

    features: list[str] = field(default_factory=lambda: FEATURES)
    """List of shape features."""

    components: int = PCA_COMPONENTS
    """Number of principal components (i.e. shape modes)."""


@dataclass
class ParametersConfigFeatureDistributions:
    """Parameter configuration for format panel data feature distributions subflow."""

    reference_model: Optional[str] = None
    """Full key for reference PCA model."""

    reference_data: Optional[str] = None
    """Full key for reference coefficients data."""

    features: list[str] = field(default_factory=lambda: FEATURES)
    """List of shape features."""

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""

    components: int = PCA_COMPONENTS
    """Number of principal components (i.e. shape modes)."""


@dataclass
class ParametersConfigModeCorrelations:
    """Parameter configuration for format panel data mode correlations subflow."""

    reference_model: Optional[str] = None
    """Full key for reference PCA model."""

    reference_data: Optional[str] = None
    """Full key for reference coefficients data."""

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""

    components: int = PCA_COMPONENTS
    """Number of principal components (i.e. shape modes)."""


@dataclass
class ParametersConfigPopulationCounts:
    """Parameter configuration for format panel data population counts subflow."""

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""

    tick: int = 0
    """Simulation tick to use for formatting population counts."""

    seeds: list[int] = field(default_factory=lambda: [0])
    """Simulation seeds to use for formatting population counts."""


@dataclass
class ParametersConfigPopulationStats:
    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""


@dataclass
class ParametersConfigShapeAverage:
    """Parameter configuration for format panel data shape average subflow."""

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""

    components: int = PCA_COMPONENTS
    """Number of principal components (i.e. shape modes)."""

    order: int = COEFFICIENT_ORDER
    """Order of the spherical harmonics coefficient parametrization."""

    scale: float = 1
    """Scaling for spherical harmonics reconstruction mesh."""


@dataclass
class ParametersConfigShapeErrors:
    """Parameter configuration for format panel data shape errors subflow."""

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""


@dataclass
class ParametersConfigShapeModes:
    """Parameter configuration for format panel data shape modes subflow."""

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""

    components: int = PCA_COMPONENTS
    """Number of principal components (i.e. shape modes)."""

    order: int = COEFFICIENT_ORDER
    """Order of the spherical harmonics coefficient parametrization."""

    delta: float = 0.5
    """Increment for shape mode map points."""


@dataclass
class ParametersConfigShapeSamples:
    """Parameter configuration for format panel data shape samples subflow."""

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""

    seed: int = 0
    """Simulation random seed to use for formatting shape samples."""

    tick: int = 0
    """Simulation tick to use for formatting shape samples."""

    indices: list[int] = field(default_factory=lambda: [0])
    """Cell indicies for shape samples."""


@dataclass
class ParametersConfigVarianceExplained:
    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""

    components: int = PCA_COMPONENTS
    """Number of principal components (i.e. shape modes)."""


@dataclass
class ParametersConfig:
    """Parameter configuration for format panel data flow."""

    panels: list[str] = field(default_factory=lambda: PANELS)
    """List of panel types to format."""

    feature_bins: ParametersConfigFeatureBins = ParametersConfigFeatureBins()
    """Parameters for format feature bins subflow."""

    feature_correlations: ParametersConfigFeatureCorrelations = (
        ParametersConfigFeatureCorrelations()
    )
    """Parameters for format feature correlations subflow."""

    feature_distributions: ParametersConfigFeatureDistributions = (
        ParametersConfigFeatureDistributions()
    )
    """Parameters for format feature distributions subflow."""

    mode_correlations: ParametersConfigModeCorrelations = ParametersConfigModeCorrelations()
    """Parameters for format mode correlations subflow."""

    population_counts: ParametersConfigPopulationCounts = ParametersConfigPopulationCounts()
    """Parameters for format population counts subflow."""

    population_stats: ParametersConfigPopulationStats = ParametersConfigPopulationStats()
    """Parameters for format population stats subflow."""

    shape_average: ParametersConfigShapeAverage = ParametersConfigShapeAverage()
    """Parameters for format shape average subflow."""

    shape_errors: ParametersConfigShapeErrors = ParametersConfigShapeErrors()
    """Parameters for format shape errors subflow."""

    shape_modes: ParametersConfigShapeModes = ParametersConfigShapeModes()
    """Parameters for format shape modes subflow."""

    shape_samples: ParametersConfigShapeSamples = ParametersConfigShapeSamples()
    """Parameters for format shape samples subflow."""

    variance_explained: ParametersConfigVarianceExplained = ParametersConfigVarianceExplained()
    """Parameters for format variance explained subflow."""


@dataclass
class ContextConfig:
    """Context configuration for format panel data flow."""

    working_location: str
    """Location for input and output files (local path or S3 bucket)."""


@dataclass
class SeriesConfig:
    """Series configuration for format panel data flow."""

    name: str
    """Name of the simulation series."""

    conditions: list[dict]
    """List of series condition dictionaries (must include unique condition "key")."""


@flow(name="format-panel-data")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    """
    Main format panel data flow.

    Calls the following subflows, if the panel is specified:

    - :py:func:`run_flow_format_feature_bins`
    - :py:func:`run_flow_format_feature_correlations`
    - :py:func:`run_flow_format_feature_distributions`
    - :py:func:`run_flow_format_mode_correlations`
    - :py:func:`run_flow_format_population_counts`
    - :py:func:`run_flow_format_population_stats`
    - :py:func:`run_flow_format_shape_average`
    - :py:func:`run_flow_format_shape_errors`
    - :py:func:`run_flow_format_shape_modes`
    - :py:func:`run_flow_format_shape_samples`
    - :py:func:`run_flow_format_variance_explained`
    """

    if "feature_bins" in parameters.panels:
        run_flow_format_feature_bins(context, series, parameters.feature_bins)

    if "feature_correlations" in parameters.panels:
        run_flow_format_feature_correlations(context, series, parameters.feature_correlations)

    if "feature_distributions" in parameters.panels:
        run_flow_format_feature_distributions(context, series, parameters.feature_distributions)

    if "mode_correlations" in parameters.panels:
        run_flow_format_mode_correlations(context, series, parameters.mode_correlations)

    if "population_counts" in parameters.panels:
        run_flow_format_population_counts(context, series, parameters.population_counts)

    if "population_stats" in parameters.panels:
        run_flow_format_population_stats(context, series, parameters.population_stats)

    if "shape_average" in parameters.panels:
        run_flow_format_shape_average(context, series, parameters.shape_average)

    if "shape_errors" in parameters.panels:
        run_flow_format_shape_errors(context, series, parameters.shape_errors)

    if "shape_modes" in parameters.panels:
        run_flow_format_shape_modes(context, series, parameters.shape_modes)

    if "shape_samples" in parameters.panels:
        run_flow_format_shape_samples(context, series, parameters.shape_samples)

    if "variance_explained" in parameters.panels:
        run_flow_format_variance_explained(context, series, parameters.variance_explained)


@flow(name="format-panel-data_format-feature-bins")
def run_flow_format_feature_bins(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigFeatureBins
) -> None:
    """Format panel data subflow for feature bins."""

    data_key = make_key(series.name, "data", "data.LOCATIONS")
    panel_key = make_key(series.name, "panels")
    keys = [condition["key"] for condition in series.conditions]

    all_feature_bins = []

    for key in keys:
        tars = {}

        for seed in parameters.seeds:
            series_key = f"{series.name}_{key}_{seed:04d}"
            tar_key = make_key(data_key, f"{series_key}.LOCATIONS.tar.xz")
            tar = load_tar(context.working_location, tar_key)
            tars[series_key] = tar

        feature_bins = extract_feature_bins(tars, parameters.tick, parameters.scale)
        feature_bins["key"] = key
        all_feature_bins.append(feature_bins)

    save_dataframe(
        context.working_location,
        make_key(panel_key, f"{series.name}.feature_bins.csv"),
        pd.concat(all_feature_bins),
        index=False,
    )


@flow(name="format-panel-data_format-feature-correlations")
def run_flow_format_feature_correlations(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigFeatureCorrelations
) -> None:
    """Format panel data subflow for feature correlations."""

    analysis_key = make_key(series.name, "analysis", "analysis.PCA")
    panel_key = make_key(series.name, "panels")
    region_key = ":".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

    features = [
        f"{feature}.{region}" if region != "DEFAULT" else feature
        for feature in parameters.features
        for region in parameters.regions
    ]

    correlations: list[dict[str, Union[str, int, float]]] = []

    for key in keys:
        # Load model.
        model_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.PCA.pkl")
        model = load_pickle(context.working_location, model_key)

        # Load dataframe.
        dataframe_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.PCA.csv")
        data = load_dataframe(context.working_location, dataframe_key)

        # Transform data into shape mode space.
        columns = data.filter(like="shcoeffs").columns
        transform = model.transform(data[columns].values)

        for feature in features:
            feature_data = data[feature]

            for component in range(parameters.components):
                component_data = transform[:, component]

                correlations.append(
                    {
                        "key": key,
                        "feature": feature,
                        "mode": component + 1,
                        "correlation": pearsonr(feature_data, component_data).statistic,
                        "correlation_symmetric": pearsonr(
                            feature_data, abs(component_data)
                        ).statistic,
                    }
                )

    save_dataframe(
        context.working_location,
        make_key(panel_key, f"{series.name}.feature_correlations.csv"),
        pd.DataFrame(correlations),
        index=False,
    )


@flow(name="format-panel-data_format-feature-distributions")
def run_flow_format_feature_distributions(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigFeatureDistributions
) -> None:
    """Format panel data subflow for feature distributions."""

    analysis_key = make_key(series.name, "analysis", "analysis.PCA")
    panel_key = make_key(series.name, "panels")
    region_key = ":".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

    columns = ["KEY", "ID", "SEED", "TICK"] + [
        f"{feature}.{region}" if region != "DEFAULT" else feature
        for feature in parameters.features
        for region in parameters.regions
    ]

    distributions: list[pd.DataFrame] = []

    if parameters.reference_model is not None and parameters.reference_data is not None:
        ref_data = load_dataframe(context.working_location, parameters.reference_data, nrows=1)
        ref_model = load_pickle(context.working_location, parameters.reference_model)
        columns = columns + [f"PC{component + 1}" for component in range(parameters.components)]

    for key in keys:
        # Load dataframe.
        dataframe_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.PCA.csv")
        data = load_dataframe(context.working_location, dataframe_key)

        # Calculate shape modes, if model is given.
        if parameters.reference_model is not None:
            transform = ref_model.transform(data[ref_data.filter(like="shcoeffs").columns].values)
            for component in range(parameters.components):
                data[f"PC{component + 1}"] = transform[:, component]

        data.drop(columns=[col for col in data if col not in columns], inplace=True)
        distributions.append(data)

    save_dataframe(
        context.working_location,
        make_key(panel_key, f"{series.name}.feature_distributions.csv"),
        pd.concat(distributions),
        index=False,
    )


@flow(name="format-panel-data_format-mode-correlations")
def run_flow_format_mode_correlations(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigModeCorrelations
) -> None:
    """Format panel data subflow for mode correlations."""

    analysis_key = make_key(series.name, "analysis", "analysis.PCA")
    panel_key = make_key(series.name, "panels")
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

    if parameters.reference_model is not None and parameters.reference_data is not None:
        keys.append("reference")
        all_models["reference"] = load_pickle(context.working_location, parameters.reference_model)
        all_data["reference"] = load_dataframe(context.working_location, parameters.reference_data)

    correlations: list[dict[str, Union[str, int, float]]] = []

    for source_key in keys:
        for target_key in keys:
            if source_key == target_key:
                continue

            # Select data sets.
            data_source = all_data[source_key]
            data_target = all_data[target_key]

            # Select models.
            model_source = all_models[source_key]
            model_target = all_models[target_key]

            # Get column order for model.
            columns_source = data_source.filter(like="shcoeffs").columns
            columns_target = data_target.filter(like="shcoeffs").columns

            # Transform the data.
            transform_source = model_source.transform(
                np.append(
                    data_source[columns_source].values,
                    data_target[columns_source].values,
                    axis=0,
                )
            )
            transform_target = model_target.transform(
                np.append(
                    data_source[columns_target].values,
                    data_target[columns_target].values,
                    axis=0,
                )
            )

            # Calculate correlations.
            correlations = correlations + [
                {
                    "source_key": source_key,
                    "target_key": target_key,
                    "source_mode": si + 1,
                    "target_mode": ti + 1,
                    "correlation": pearsonr(
                        transform_source[:, si], transform_target[:, ti]
                    ).statistic,
                }
                for si in range(parameters.components)
                for ti in range(parameters.components)
            ]

    save_dataframe(
        context.working_location,
        make_key(panel_key, f"{series.name}.mode_correlations.csv"),
        pd.DataFrame(correlations),
        index=False,
    )


@flow(name="format-panel-data_format-population-counts")
def run_flow_format_population_counts(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigPopulationCounts
) -> None:
    """Format panel data subflow for population counts."""

    results_key = make_key(series.name, "results")
    analysis_key = make_key(series.name, "analysis", "analysis.PCA")
    panel_key = make_key(series.name, "panels")
    region_key = ":".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

    counts = []

    for key in keys:
        data_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.PCA.csv")
        data = load_dataframe(context.working_location, data_key, usecols=["TICK", "SEED"])
        groups = data[data["TICK"] == parameters.tick].groupby("SEED")

        for seed in parameters.seeds:
            data_key = make_key(results_key, f"{series.name}_{key}_{seed:04d}.csv")
            results = load_dataframe(context.working_location, data_key, usecols=["TICK"])

            counts.append(
                {
                    "key": key,
                    "seed": seed,
                    "total_count": len(results[results["TICK"] == parameters.tick]),
                    "filter_count": len(groups.get_group(seed)),
                }
            )

    save_dataframe(
        context.working_location,
        make_key(panel_key, f"{series.name}.population_counts.csv"),
        pd.DataFrame(counts),
        index=False,
    )


@flow(name="format-panel-data_format-population-stats")
def run_flow_format_population_stats(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigPopulationStats
) -> None:
    """Format panel data subflow for population stats."""

    analysis_key = make_key(series.name, "analysis", "analysis.STATS")
    panel_key = make_key(series.name, "panels")
    region_key = ":".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

    stats: list[pd.DataFrame] = []

    for key in keys:
        data_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.STATS.csv")
        data = load_dataframe(context.working_location, data_key)

        sample_data = data[~data["SAMPLE"].isna()].drop(columns=["TICK", "time"])
        sample_data["key"] = key

        stats.append(sample_data)

    save_dataframe(
        context.working_location,
        make_key(panel_key, f"{series.name}.population_stats.csv"),
        pd.concat(stats),
        index=False,
    )


@flow(name="format-panel-data_format-shape-average")
def run_flow_format_shape_average(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigShapeAverage
) -> None:
    """
    Format panel data subflow for shape average.

    Find the cell closest to the average shape. Extract original mesh slice and
    extent projections. Created reconstructed mesh and extract mesh slice and
    extent projections.
    """

    logger = get_run_logger()

    analysis_key = make_key(series.name, "analysis", "analysis.PCA")
    data_key = make_key(series.name, "data", "data.LOCATIONS")
    panel_key = make_key(series.name, "panels")
    region_key = ":".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

    shape_average: dict[str, dict] = {
        "original": {},
        "reconstructed": {},
    }

    for key in keys:
        # Load model.
        model_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.PCA.pkl")
        model = load_pickle(context.working_location, model_key)

        # Load dataframe.
        dataframe_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.PCA.csv")
        data = load_dataframe(context.working_location, dataframe_key)

        # Transform data into shape mode space.
        columns = data.filter(like="shcoeffs").columns
        transform = model.transform(data[columns].values)

        # Select the cell closest to average.
        distance, index = KDTree(transform).query([0] * parameters.components)
        selected = data.iloc[index, :]
        logger.info(
            "[ %s ] seed [ %d ] tick [ %d ] cell [ %d ] with distance [ %.2f ]",
            key,
            selected["SEED"],
            selected["TICK"],
            selected["ID"],
            distance,
        )

        # Get the matching location for the selected cell.
        series_key = f"{series.name}_{key}_{selected['SEED']:04d}"
        tar_key = make_key(data_key, f"{series_key}.LOCATIONS.tar.xz")
        tar = load_tar(context.working_location, tar_key)

        # Load matching location voxels.
        locations = extract_tick_json(tar, series_key, selected["TICK"], "LOCATIONS")
        location = next(location for location in locations if location["id"] == selected["ID"])
        voxels = get_location_voxels(location)
        array = make_voxels_array(voxels)

        # Create original mesh and get projections.
        original_mesh = construct_mesh_from_array(array, array)
        shape_average["original"][key] = extract_mesh_projections(original_mesh)

        # Create reconstructed mesh and get projections.
        reconstructed_mesh = construct_mesh_from_coeffs(
            selected, parameters.order, scale=parameters.scale
        )
        shape_average["reconstructed"][key] = extract_mesh_projections(reconstructed_mesh)

    save_json(
        context.working_location,
        make_key(panel_key, f"{series.name}.shape_average.json"),
        shape_average,
    )


@flow(name="format-panel-data_format-shape-errors")
def run_flow_format_shape_errors(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigShapeErrors
) -> None:
    """Format panel data subflow for shape errors."""

    analysis_key = make_key(series.name, "analysis", "analysis.PCA")
    panel_key = make_key(series.name, "panels")
    region_key = ":".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

    columns = ["KEY", "ID", "SEED", "TICK"] + [
        f"mse.{region}" if region != "DEFAULT" else "mse" for region in parameters.regions
    ]

    errors: list[pd.DataFrame] = []

    for key in keys:
        data_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.PCA.csv")
        data = load_dataframe(context.working_location, data_key)
        errors.append(data[columns])

    save_dataframe(
        context.working_location,
        make_key(panel_key, f"{series.name}.shape_errors.csv"),
        pd.concat(errors),
        index=False,
    )


@flow(name="format-panel-data_format-shape-modes")
def run_flow_format_shape_modes(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigShapeModes
) -> None:
    """
    Format panel data subflow for shape modes.

    Extract shape modes from PCAs as dictionaries of svg paths for each map
    point and projection. Consolidate shape modes from keys into single json.
    """

    analysis_key = make_key(series.name, "analysis", "analysis.PCA")
    panel_key = make_key(series.name, "panels")
    region_key = ":".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

    shape_modes: dict[str, dict] = {}

    for key in keys:
        shape_modes[key] = {region: [] for region in parameters.regions}

        # Load model.
        model_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.PCA.pkl")
        model = load_pickle(context.working_location, model_key)

        # Load dataframe.
        dataframe_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.PCA.csv")
        data = load_dataframe(context.working_location, dataframe_key)

        # Extract shape modes.
        shape_modes[key] = extract_shape_modes(
            model,
            data,
            parameters.components,
            parameters.regions,
            parameters.order,
            parameters.delta,
        )

    save_json(
        context.working_location,
        make_key(panel_key, f"{series.name}.shape_modes.json"),
        shape_modes,
    )


@flow(name="format-panel-data_format-shape-samples")
def run_flow_format_shape_samples(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigShapeSamples
) -> None:
    """
    Format panel data subflow for shape samples.

    Extract sample cell shapes from specified simulations. Construct wireframes
    from the cell shape mesh.
    """

    data_key = make_key(series.name, "data", "data.LOCATIONS")
    panel_key = make_key(series.name, "panels")
    keys = [condition["key"] for condition in series.conditions]

    shape_samples: dict[str, dict] = {}

    for key in keys:
        shape_samples[key] = {region: [] for region in parameters.regions}

        # Load location data.
        series_key = f"{series.name}_{key}_{parameters.seed:04d}"
        tar_key = make_key(data_key, f"{series_key}.LOCATIONS.tar.xz")
        tar = load_tar(context.working_location, tar_key)
        locations = extract_tick_json(tar, series_key, parameters.tick, "LOCATIONS")

        for index in parameters.indices:
            location = locations[index]

            for region in parameters.regions:
                voxels = get_location_voxels(location)
                array = make_voxels_array(voxels)

                if region != "DEFAULT":
                    region_voxels = get_location_voxels(locations[index], region)
                    region_array = make_voxels_array(region_voxels, reference=voxels)
                    mesh = construct_mesh_from_array(region_array, array)
                else:
                    mesh = construct_mesh_from_array(array, array)

                shape_samples[key][region].append(extract_mesh_wireframe(mesh))

    save_json(
        context.working_location,
        make_key(panel_key, f"{series.name}.shape_samples.json"),
        shape_samples,
    )


@flow(name="format-panel-data_format-variance-explained")
def run_flow_format_variance_explained(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigVarianceExplained
) -> None:
    """Format panel data subflow for variance explained."""

    analysis_key = make_key(series.name, "analysis", "analysis.PCA")
    panel_key = make_key(series.name, "panels")
    region_key = ":".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

    variance = []

    for key in keys:
        model_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.PCA.pkl")
        model = load_pickle(context.working_location, model_key)

        variance.append(
            pd.DataFrame(
                {
                    "key": [key] * parameters.components,
                    "mode": range(1, parameters.components + 1),
                    "variance": model.explained_variance_ratio_,
                }
            )
        )

    save_dataframe(
        context.working_location,
        make_key(panel_key, f"{series.name}.variance_explained.csv"),
        pd.concat(variance),
        index=False,
    )
