"""
Workflow for grouping cell shapes.

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
    │       ├── (name)_(key)_(regions).STATS.csv
    │       ├── (name)_(key)_(regions).STATS.csv
    │       ├── ...
    │       └── (name)_(key)_(regions).STATS.csv
    ├── data
    │   └── data.LOCATIONS
    │       ├── (name)_(key)_(seed).LOCATIONS.tar.xz
    │       ├── (name)_(key)_(seed).LOCATIONS.tar.xz
    │       ├── ...
    │       └── (name)_(key)_(seed).LOCATIONS.tar.xz
    └── groups
        └── groups.SHAPES
            ├── (name).feature_correlations.(key).json
            ├── (name).feature_correlations.(key).json
            ├── ...
            ├── (name).feature_correlations.(key).json
            ├── (name).feature_correlations.(key).(mode).(property).csv
            ├── (name).feature_correlations.(key).(mode).(property).csv
            ├── ...
            ├── (name).feature_correlations.(key).(mode).(property).csv
            ├── (name).feature_distributions.(feature).json
            ├── (name).feature_distributions.(feature).json
            ├── ...
            ├── (name).feature_distributions.(feature).json
            ├── (name).mode_correlations.csv
            ├── (name).population_counts.csv
            ├── (name).population_stats.json
            ├── (name).shape_average.(key).(projection).json
            ├── (name).shape_average.(key).(projection).json
            ├── ...
            ├── (name).shape_average.(key).(projection).json
            ├── (name).shape_errors.json
            ├── (name).shape_modes.(key).(region).(mode).(projection).json
            ├── (name).shape_modes.(key).(region).(mode).(projection).json
            ├── ...
            ├── (name).shape_modes.(key).(region).(mode).(projection).json
            ├── (name).shape_samples.json
            └── (name).variance_explained.csv

Different groups use inputs from the **data/data.LOCATIONS**,
**analysis/analysis.PCA**, and **analysis/analysis.STATS** directories.
Grouped data is saved to the **groups/groups.SHAPES** directory.

Different groups can be visualized using the corresponding plotting workflow or
loaded into alternative tools.
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
from arcade_collection.output import extract_tick_json, get_location_voxels
from io_collection.keys import make_key
from io_collection.load import load_dataframe, load_pickle, load_tar
from io_collection.save import save_dataframe, save_json
from prefect import flow, get_run_logger
from scipy.spatial import KDTree
from scipy.stats import pearsonr

from cell_abm_pipeline.flows.analyze_cell_shapes import PCA_COMPONENTS
from cell_abm_pipeline.flows.calculate_coefficients import COEFFICIENT_ORDER
from cell_abm_pipeline.tasks import bin_to_hex, calculate_data_bins

GROUPS: list[str] = [
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

PROPERTIES: list[str] = [
    "volume",
    "height",
    "area",
    "axis_major_length",
    "axis_minor_length",
    "eccentricity",
    "perimeter",
]

PROJECTIONS: list[str] = [
    "top",
    "side1",
    "side2",
]

BOUNDS: dict[str, list] = {
    "volume.DEFAULT": [0, 5000],
    "volume.NUCLEUS": [0, 1500],
    "height.DEFAULT": [0, 20],
    "height.NUCLEUS": [0, 20],
    "PC1": [-50, 50],
    "PC2": [-50, 50],
    "PC3": [-50, 50],
    "PC4": [-50, 50],
    "PC5": [-50, 50],
    "PC6": [-50, 50],
    "PC7": [-50, 50],
    "PC8": [-50, 50],
}

BANDWIDTH: dict[str, float] = {
    "volume.DEFAULT": 100,
    "volume.NUCLEUS": 50,
    "height.DEFAULT": 1,
    "height.NUCLEUS": 1,
    "PC1": 5,
    "PC2": 5,
    "PC3": 5,
    "PC4": 5,
    "PC5": 5,
    "PC6": 5,
    "PC7": 5,
    "PC8": 5,
}


@dataclass
class ParametersConfigFeatureCorrelations:
    """Parameter configuration for group cell shapes subflow - feature correlations."""

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""

    properties: list[str] = field(default_factory=lambda: PROPERTIES)
    """List of shape properties."""

    components: int = PCA_COMPONENTS
    """Number of principal components (i.e. shape modes)."""

    include_bins: bool = False
    """True if correlations are binned, False otherwise"""


@dataclass
class ParametersConfigFeatureDistributions:
    """Parameter configuration for group cell shapes subflow - feature distributions."""

    reference_model: Optional[str] = None
    """Full key for reference PCA model."""

    reference_data: Optional[str] = None
    """Full key for reference coefficients data."""

    properties: list[str] = field(default_factory=lambda: PROPERTIES)
    """List of shape properties."""

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""

    components: int = PCA_COMPONENTS
    """Number of principal components (i.e. shape modes)."""

    bounds: dict[str, list] = field(default_factory=lambda: BOUNDS)
    """Bounds for metric distributions."""

    bandwidth: dict[str, float] = field(default_factory=lambda: BANDWIDTH)
    """Bandwidths for metric distributions."""


@dataclass
class ParametersConfigModeCorrelations:
    """Parameter configuration for group cell shapes subflow - mode correlations."""

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
    """Parameter configuration for group cell shapes subflow - population counts."""

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""

    tick: int = 0
    """Simulation tick to use for grouping population counts."""

    seeds: list[int] = field(default_factory=lambda: [0])
    """Simulation seeds to use for grouping population counts."""


@dataclass
class ParametersConfigPopulationStats:
    """Parameter configuration for group cell shapes subflow - population stats."""

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""


@dataclass
class ParametersConfigShapeAverage:
    """Parameter configuration for group cell shapes subflow - shape average."""

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""

    components: int = PCA_COMPONENTS
    """Number of principal components (i.e. shape modes)."""

    order: int = COEFFICIENT_ORDER
    """Order of the spherical harmonics coefficient parametrization."""

    scale: float = 1
    """Scaling for spherical harmonics reconstruction mesh."""

    projections: list[str] = field(default_factory=lambda: PROJECTIONS)
    """List of shape projections."""


@dataclass
class ParametersConfigShapeErrors:
    """Parameter configuration for group cell shapes subflow - shape errors."""

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""


@dataclass
class ParametersConfigShapeModes:
    """Parameter configuration for group cell shapes subflow - shape modes."""

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""

    components: int = PCA_COMPONENTS
    """Number of principal components (i.e. shape modes)."""

    order: int = COEFFICIENT_ORDER
    """Order of the spherical harmonics coefficient parametrization."""

    delta: float = 0.5
    """Increment for shape mode map points."""

    projections: list[str] = field(default_factory=lambda: PROJECTIONS)
    """List of shape projections."""


@dataclass
class ParametersConfigShapeSamples:
    """Parameter configuration for group cell shapes subflow - shape samples."""

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""

    seed: int = 0
    """Simulation random seed to use for grouping shape samples."""

    tick: int = 0
    """Simulation tick to use for grouping shape samples."""

    indices: list[int] = field(default_factory=lambda: [0])
    """Cell indices for shape samples."""


@dataclass
class ParametersConfigVarianceExplained:
    """Parameter configuration for group cell shapes subflow - variance explained."""

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""

    components: int = PCA_COMPONENTS
    """Number of principal components (i.e. shape modes)."""


@dataclass
class ParametersConfig:
    """Parameter configuration for group cell shapes flow."""

    groups: list[str] = field(default_factory=lambda: GROUPS)
    """List of cell shape groups."""

    feature_correlations: ParametersConfigFeatureCorrelations = (
        ParametersConfigFeatureCorrelations()
    )
    """Parameters for group feature correlations subflow."""

    feature_distributions: ParametersConfigFeatureDistributions = (
        ParametersConfigFeatureDistributions()
    )
    """Parameters for group feature distributions subflow."""

    mode_correlations: ParametersConfigModeCorrelations = ParametersConfigModeCorrelations()
    """Parameters for group mode correlations subflow."""

    population_counts: ParametersConfigPopulationCounts = ParametersConfigPopulationCounts()
    """Parameters for group population counts subflow."""

    population_stats: ParametersConfigPopulationStats = ParametersConfigPopulationStats()
    """Parameters for group population stats subflow."""

    shape_average: ParametersConfigShapeAverage = ParametersConfigShapeAverage()
    """Parameters for group shape average subflow."""

    shape_errors: ParametersConfigShapeErrors = ParametersConfigShapeErrors()
    """Parameters for group shape errors subflow."""

    shape_modes: ParametersConfigShapeModes = ParametersConfigShapeModes()
    """Parameters for group shape modes subflow."""

    shape_samples: ParametersConfigShapeSamples = ParametersConfigShapeSamples()
    """Parameters for group shape samples subflow."""

    variance_explained: ParametersConfigVarianceExplained = ParametersConfigVarianceExplained()
    """Parameters for group variance explained subflow."""


@dataclass
class ContextConfig:
    """Context configuration for group cell shapes flow."""

    working_location: str
    """Location for input and output files (local path or S3 bucket)."""


@dataclass
class SeriesConfig:
    """Series configuration for group cell shapes flow."""

    name: str
    """Name of the simulation series."""

    conditions: list[dict]
    """List of series condition dictionaries (must include unique condition "key")."""


@flow(name="group-cell-shapes")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    """
    Main group cell shapes flow.

    Calls the following subflows, if the group is specified:

    - :py:func:`run_flow_group_feature_correlations`
    - :py:func:`run_flow_group_feature_distributions`
    - :py:func:`run_flow_group_mode_correlations`
    - :py:func:`run_flow_group_population_counts`
    - :py:func:`run_flow_group_population_stats`
    - :py:func:`run_flow_group_shape_average`
    - :py:func:`run_flow_group_shape_errors`
    - :py:func:`run_flow_group_shape_modes`
    - :py:func:`run_flow_group_shape_samples`
    - :py:func:`run_flow_group_variance_explained`
    """

    if "feature_correlations" in parameters.groups:
        run_flow_group_feature_correlations(context, series, parameters.feature_correlations)

    if "feature_distributions" in parameters.groups:
        run_flow_group_feature_distributions(context, series, parameters.feature_distributions)

    if "mode_correlations" in parameters.groups:
        run_flow_group_mode_correlations(context, series, parameters.mode_correlations)

    if "population_counts" in parameters.groups:
        run_flow_group_population_counts(context, series, parameters.population_counts)

    if "population_stats" in parameters.groups:
        run_flow_group_population_stats(context, series, parameters.population_stats)

    if "shape_average" in parameters.groups:
        run_flow_group_shape_average(context, series, parameters.shape_average)

    if "shape_errors" in parameters.groups:
        run_flow_group_shape_errors(context, series, parameters.shape_errors)

    if "shape_modes" in parameters.groups:
        run_flow_group_shape_modes(context, series, parameters.shape_modes)

    if "shape_samples" in parameters.groups:
        run_flow_group_shape_samples(context, series, parameters.shape_samples)

    if "variance_explained" in parameters.groups:
        run_flow_group_variance_explained(context, series, parameters.variance_explained)


@flow(name="group-cell-shapes_group-feature-correlations")
def run_flow_group_feature_correlations(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigFeatureCorrelations
) -> None:
    """Group cell shapes subflow for feature correlations."""

    analysis_key = make_key(series.name, "analysis", "analysis.PCA")
    group_key = make_key(series.name, "groups", "groups.SHAPES")
    region_key = ":".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

    for key in keys:
        feature_key = f"{series.name}.feature_correlations.{key}"
        correlations: list[dict[str, Union[str, float]]] = []

        # Load model.
        model_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.PCA.pkl")
        model = load_pickle(context.working_location, model_key)

        # Load dataframe.
        dataframe_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.PCA.csv")
        data = load_dataframe(context.working_location, dataframe_key)

        # Transform data into shape mode space.
        columns = data.filter(like="shcoeffs").columns
        transform = model.transform(data[columns].values)

        for component in range(parameters.components):
            mode_key = f"PC{component + 1}"
            component_data = transform[:, component]

            for prop in parameters.properties:
                for region in parameters.regions:
                    prop_key = f"{prop.upper()}.{region}"
                    prop_data = data[f"{prop}.{region}".replace(".DEFAULT", "")]

                    slope, intercept = np.polyfit(component_data, prop_data, 1)

                    correlations.append(
                        {
                            "mode": mode_key,
                            "property": prop.upper(),
                            "region": region,
                            "correlation": pearsonr(prop_data, component_data).statistic,
                            "correlation_symmetric": pearsonr(
                                prop_data, abs(component_data)
                            ).statistic,
                            "slope": slope,
                            "intercept": intercept,
                        }
                    )

                    if not parameters.include_bins:
                        continue

                    bins = bin_to_hex(
                        component_data, prop_data, [0] * len(prop_data), scale=0.05, rescale=True
                    )
                    bins_df = pd.DataFrame(
                        [[x, y, np.sum(v)] for (x, y), v in bins.items()], columns=["x", "y", "v"]
                    )

                    save_dataframe(
                        context.working_location,
                        make_key(group_key, f"{feature_key}.{mode_key}.{prop_key}.csv"),
                        bins_df,
                        index=False,
                    )

        save_dataframe(
            context.working_location,
            make_key(group_key, f"{series.name}.feature_correlations.{key}.csv"),
            pd.DataFrame(correlations),
            index=False,
        )


@flow(name="group-cell-shapes_group-feature-distributions")
def run_flow_group_feature_distributions(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigFeatureDistributions
) -> None:
    """Group cell shapes subflow for feature distributions."""

    analysis_key = make_key(series.name, "analysis", "analysis.PCA")
    group_key = make_key(series.name, "groups", "groups.SHAPES")
    region_key = ":".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

    features = [
        f"{prop}.{region}" for prop in parameters.properties for region in parameters.regions
    ]

    if parameters.reference_model is not None and parameters.reference_data is not None:
        ref_data = load_dataframe(context.working_location, parameters.reference_data, nrows=1)
        ref_model = load_pickle(context.working_location, parameters.reference_model)
        features = features + [f"PC{component + 1}" for component in range(parameters.components)]

    distribution_bins: dict[str, dict] = {feature: {} for feature in features}
    distribution_means: dict[str, dict] = {feature: {} for feature in features}
    distribution_stdevs: dict[str, dict] = {feature: {} for feature in features}

    for key in keys:
        # Load dataframe.
        dataframe_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.PCA.csv")
        data = load_dataframe(context.working_location, dataframe_key)

        # Calculate shape modes, if model is given.
        if parameters.reference_model is not None:
            transform = ref_model.transform(data[ref_data.filter(like="shcoeffs").columns].values)
            for component in range(parameters.components):
                data[f"PC{component + 1}"] = transform[:, component]

        for feature in features:
            feature_data = data[feature.replace(".DEFAULT", "")].values

            bounds = (parameters.bounds[feature][0], parameters.bounds[feature][1])
            bandwidth = parameters.bandwidth[feature]

            distribution_means[feature][key] = np.mean(feature_data)
            distribution_stdevs[feature][key] = np.std(feature_data, ddof=1)
            distribution_bins[feature][key] = calculate_data_bins(feature_data, bounds, bandwidth)

    for feature, distribution in distribution_bins.items():
        distribution["*"] = {
            "bandwidth": parameters.bandwidth[feature],
            "means": distribution_means[feature],
            "stdevs": distribution_stdevs[feature],
        }

        save_json(
            context.working_location,
            make_key(group_key, f"{series.name}.feature_distributions.{feature.upper()}.json"),
            distribution,
        )


@flow(name="group-cell-shapes_group-mode-correlations")
def run_flow_group_mode_correlations(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigModeCorrelations
) -> None:
    """Group cell shapes subflow for mode correlations."""

    analysis_key = make_key(series.name, "analysis", "analysis.PCA")
    group_key = make_key(series.name, "groups", "groups.SHAPES")
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
                    "source_mode": f"PC{si + 1}",
                    "target_mode": f"PC{ti + 1}",
                    "correlation": pearsonr(
                        transform_source[:, si], transform_target[:, ti]
                    ).statistic,
                }
                for si in range(parameters.components)
                for ti in range(parameters.components)
            ]

    save_dataframe(
        context.working_location,
        make_key(group_key, f"{series.name}.mode_correlations.csv"),
        pd.DataFrame(correlations),
        index=False,
    )


@flow(name="group-cell-shapes_group-population-counts")
def run_flow_group_population_counts(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigPopulationCounts
) -> None:
    """Group cell shapes subflow for population counts."""

    analysis_key = make_key(series.name, "analysis", "analysis.PCA")
    group_key = make_key(series.name, "groups", "groups.SHAPES")
    region_key = ":".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

    counts = []

    for key in keys:
        data_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.PCA.csv")
        data = load_dataframe(context.working_location, data_key, usecols=["TICK", "SEED"])
        groups = data[data["TICK"] == parameters.tick].groupby("SEED")

        for seed in parameters.seeds:
            counts.append(
                {
                    "key": key,
                    "seed": seed,
                    "count": len(groups.get_group(seed)),
                }
            )

    save_dataframe(
        context.working_location,
        make_key(group_key, f"{series.name}.population_counts.csv"),
        pd.DataFrame(counts),
        index=False,
    )


@flow(name="group-cell-shapes_group-population-stats")
def run_flow_group_population_stats(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigPopulationStats
) -> None:
    """Group cell shapes subflow for population stats."""

    analysis_key = make_key(series.name, "analysis", "analysis.STATS")
    group_key = make_key(series.name, "groups", "groups.SHAPES")
    region_key = ":".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

    stats: dict[str, dict] = {key: {} for key in keys}

    for key in keys:
        data_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.STATS.csv")
        data = load_dataframe(context.working_location, data_key)

        for feature, group in data[~data["SAMPLE"].isna()].groupby("FEATURE"):
            feature_name = feature.replace("_", "")
            feature_name = "volume.DEFAULT" if feature == "volume" else feature_name
            feature_name = "height.DEFAULT" if feature == "height" else feature_name

            stats[key][feature_name.upper()] = {
                "mean": group["KS_STATISTIC"].mean(),
                "std": group["KS_STATISTIC"].std(ddof=1),
            }

    save_json(
        context.working_location,
        make_key(group_key, f"{series.name}.population_stats.json"),
        stats,
    )


@flow(name="group-cell-shapes_group-shape-average")
def run_flow_group_shape_average(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigShapeAverage
) -> None:
    """
    Group cell shapes subflow for shape average.

    Find the cell closest to the average shape. Extract original mesh slice and
    extent projections. Created reconstructed mesh and extract mesh slice and
    extent projections.
    """

    logger = get_run_logger()

    analysis_key = make_key(series.name, "analysis", "analysis.PCA")
    data_key = make_key(series.name, "data", "data.LOCATIONS")
    group_key = make_key(series.name, "groups", "groups.SHAPES")
    region_key = ":".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

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
        original_mesh_projections = extract_mesh_projections(original_mesh)

        # Create reconstructed mesh and get projections.
        reconstructed_mesh = construct_mesh_from_coeffs(
            selected, parameters.order, scale=parameters.scale
        )
        reconstructed_mesh_projections = extract_mesh_projections(reconstructed_mesh)

        # Save json for each projection.
        for projection in parameters.projections:
            shape_average: dict[str, dict] = {
                "original_slice": original_mesh_projections[f"{projection}_slice"],
                "original_extent": original_mesh_projections[f"{projection}_extent"],
                "reconstructed_slice": reconstructed_mesh_projections[f"{projection}_slice"],
            }

            save_json(
                context.working_location,
                make_key(group_key, f"{series.name}.shape_average.{key}.{projection.upper()}.json"),
                shape_average,
            )


@flow(name="group-cell-shapes_group-shape-errors")
def run_flow_group_shape_errors(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigShapeErrors
) -> None:
    """Group cell shapes subflow for shape errors."""

    analysis_key = make_key(series.name, "analysis", "analysis.PCA")
    group_key = make_key(series.name, "groups", "groups.SHAPES")
    region_key = ":".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

    errors: dict[str, dict] = {key: {} for key in keys}

    for key in keys:
        data_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.PCA.csv")
        data = load_dataframe(context.working_location, data_key)

        for region in parameters.regions:
            errors[key][region] = {
                "mean": data[f"mse.{region}".replace(".DEFAULT", "")].mean(),
                "std": data[f"mse.{region}".replace(".DEFAULT", "")].std(ddof=1),
            }

    save_json(
        context.working_location,
        make_key(group_key, f"{series.name}.shape_errors.json"),
        errors,
    )


@flow(name="group-cell-shapes_group-shape-modes")
def run_flow_group_shape_modes(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigShapeModes
) -> None:
    """
    Group cell shapes subflow for shape modes.

    Extract shape modes from PCAs as dictionaries of svg paths for each map
    point and projection. Consolidate shape modes from keys into single json.
    """

    analysis_key = make_key(series.name, "analysis", "analysis.PCA")
    group_key = make_key(series.name, "groups", "groups.SHAPES")
    region_key = ":".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

    projections = ["top", "side1", "side2"]

    for key in keys:
        # Load model.
        model_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.PCA.pkl")
        model = load_pickle(context.working_location, model_key)

        # Load dataframe.
        dataframe_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.PCA.csv")
        data = load_dataframe(context.working_location, dataframe_key)

        # Extract shape modes.
        shape_modes = extract_shape_modes(
            model,
            data,
            parameters.components,
            parameters.regions,
            parameters.order,
            parameters.delta,
        )

        for region in parameters.regions:
            shape_mode_projections: dict[str, list] = {
                f"PC{component + 1}.{projection}": []
                for component in range(parameters.components)
                for projection in parameters.projections
            }

            for shape_mode in shape_modes[region]:
                for projection in parameters.projections:
                    shape_mode_projections[f"PC{shape_mode['mode']}.{projection}"].append(
                        {
                            "point": shape_mode["point"],
                            "projection": shape_mode["projections"][f"{projection}_slice"],
                        }
                    )

            for projection_key, projections in shape_mode_projections.items():
                save_json(
                    context.working_location,
                    make_key(
                        group_key,
                        f"{series.name}.shape_modes.{key}.{region}.{projection_key.upper()}.json",
                    ).replace("..", "."),
                    projections,
                )


@flow(name="group-cell-shapes_group-shape-samples")
def run_flow_group_shape_samples(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigShapeSamples
) -> None:
    """
    Group cell shapes subflow for shape samples.

    Extract sample cell shapes from specified simulations. Construct wireframes
    from the cell shape mesh.
    """

    data_key = make_key(series.name, "data", "data.LOCATIONS")
    group_key = make_key(series.name, "groups", "groups.SHAPES")
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
        make_key(group_key, f"{series.name}.shape_samples.json"),
        shape_samples,
    )


@flow(name="group-cell-shapes_group-variance-explained")
def run_flow_group_variance_explained(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfigVarianceExplained
) -> None:
    """Group cell shapes subflow for variance explained."""

    analysis_key = make_key(series.name, "analysis", "analysis.PCA")
    group_key = make_key(series.name, "groups", "groups.SHAPES")
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
                    "mode": [f"PC{i}" for i in range(1, parameters.components + 1)],
                    "variance": model.explained_variance_ratio_,
                }
            )
        )

    save_dataframe(
        context.working_location,
        make_key(group_key, f"{series.name}.variance_explained.csv"),
        pd.concat(variance),
        index=False,
    )
