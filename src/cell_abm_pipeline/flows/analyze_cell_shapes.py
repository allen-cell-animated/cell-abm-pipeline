"""
Workflow for analyzing cell shapes.

Working location structure:

.. code-block:: bash

    (name)
    ├── analysis
    │   ├── analysis.COEFFICIENTS
    │   │   ├── (name)_(key)_(seed)_(region).COEFFICIENTS.csv
    │   │   └── (name)_(key)_(seed)_(region).COEFFICIENTS.tar.xz
    │   ├── analysis.PROPERTIES
    │   │   ├── (name)_(key)_(seed)_(region).PROPERTIES.csv
    │   │   └── (name)_(key)_(seed)_(region).PROPERTIES.tar.xz
    │   ├── analysis.PCA
    │   │   └── (name)_(key)_(regions).PCA.pkl
    │   ├── analysis.SHAPES
    │   │   └── (name)_(key)_(regions).SHAPES.csv
    │   └── analysis.STATISTICS
    │       └── (name)_(key)_(regions).STATISTICS.csv
    └── results
        └── (name)_(key)_(seed).csv

Data from the **results**, **analysis.COEFFICIENTS**, and (optionally) the
**analysis.PROPERTIES** directories are processed into the **analysis.SHAPES**
directory.
PCA models are saved to the **analysis.PCA** directory.
Statistical analysis is saved to the **analysis.STATISTICS** directory.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
from abm_shape_collection import (
    calculate_feature_statistics,
    calculate_shape_statistics,
    fit_pca_model,
)
from arcade_collection.output import convert_model_units
from io_collection.keys import check_key, make_key
from io_collection.load import load_dataframe, load_pickle
from io_collection.save import save_dataframe, save_pickle
from prefect import flow
from prefect.tasks import task_input_hash

OPTIONS = {
    "cache_result_in_memory": False,
    "cache_key_fn": task_input_hash,
    "cache_expiration": timedelta(hours=12),
}

PCA_COMPONENTS = 8

INDEX_COLUMNS = ["KEY", "ID", "SEED", "TICK"]

VALID_PHASES = ["PROLIFERATIVE_G1", "PROLIFERATIVE_S", "PROLIFERATIVE_G2"]


@dataclass
class ParametersConfig:
    """Parameter configuration for analyze cell shapes flow."""

    reference: Optional[dict] = None
    """Dictionary of keys for reference data and model for statistics."""

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""

    components: int = PCA_COMPONENTS
    """Number of principal components (i.e. shape modes)."""

    ds: float = 1.0
    """Spatial scaling in units/um."""

    dt: float = 1.0
    """Temporal scaling in hours/tick."""

    valid_phases: list[str] = field(default_factory=lambda: VALID_PHASES)
    """Valid phases for processing cell shapes."""

    valid_ticks: list[int] = field(default_factory=lambda: [0])
    """Valid ticks for processing cell shapes."""

    sample_replicates: int = 100
    """Number of replicates for calculating stats with sampling."""

    sample_size: int = 100
    """Sample size for each tick for calculating stats with sampling."""

    outlier: Optional[float] = None
    """Standard deviation threshold for outliers."""


@dataclass
class ContextConfig:
    """Context configuration for analyze cell shapes flow."""

    working_location: str
    """Location for input and output files (local path or S3 bucket)."""


@dataclass
class SeriesConfig:
    """Series configuration for analyze cell shapes flow."""

    name: str
    """Name of the simulation series."""

    seeds: list[int]
    """List of series random seeds."""

    conditions: list[dict]
    """List of series condition dictionaries (must include unique condition "key")."""


@flow(name="analyze-cell-shapes")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    """
    Main analyze cell shapes flow.

    Calls the following subflows, in order:

    1. :py:func:`run_flow_process_data`
    2. :py:func:`run_flow_fit_model`
    3. :py:func:`run_flow_analyze_stats`
    """

    run_flow_process_data(context, series, parameters)

    run_flow_fit_model(context, series, parameters)

    run_flow_analyze_stats(context, series, parameters)


@flow(name="analyze-cell-shapes_process-data")
def run_flow_process_data(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    """
    Analyze cell shapes subflow for processing data.

    Process spherical harmonics coefficients and parsed simulation results and
    compile into a single dataframe that can used for PCA. If the combined data
    already exists for a given key, that key is skipped.
    """

    results_path_key = make_key(series.name, "results")
    coeffs_path_key = make_key(series.name, "analysis", "analysis.COEFFICIENTS")
    props_path_key = make_key(series.name, "analysis", "analysis.PROPERTIES")
    shapes_path_key = make_key(series.name, "analysis", "analysis.SHAPES")
    region_key = "_".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

    for key in keys:
        data_key = make_key(shapes_path_key, f"{series.name}_{key}_{region_key}.SHAPES.csv")

        if check_key(context.working_location, data_key):
            continue

        all_results = []
        all_coeffs = []
        all_props = []

        for seed in series.seeds:
            coeffs = None
            props = None

            # Load parsed results
            results_key = make_key(results_path_key, f"{series.name}_{key}_{seed:04d}.csv")
            results = load_dataframe(context.working_location, results_key)
            results["KEY"] = key
            results["SEED"] = seed
            results.set_index(INDEX_COLUMNS, inplace=True)
            all_results.append(results)

            for region in parameters.regions:
                # Load coefficients for region.
                coeffs_key = make_key(
                    coeffs_path_key, f"{series.name}_{key}_{seed:04d}_{region}.COEFFICIENTS.csv"
                )
                coeffs_key = coeffs_key.replace("_DEFAULT", "")
                region_coeffs = load_dataframe(
                    context.working_location, coeffs_key, converters={"KEY": str}
                )
                region_coeffs.set_index(INDEX_COLUMNS, inplace=True)

                if coeffs is None:
                    coeffs = region_coeffs
                    if region != "DEFAULT":
                        coeffs = coeffs.add_suffix(f".{region}")
                else:
                    coeffs = coeffs.join(region_coeffs, on=INDEX_COLUMNS, rsuffix=f".{region}")

                # Load properties for region (if it exists).
                props_key = make_key(
                    props_path_key, f"{series.name}_{key}_{seed:04d}_{region}.PROPERTIES.csv"
                )
                props_key = props_key.replace("_DEFAULT", "")

                if not check_key(context.working_location, props_key):
                    props = None
                    continue

                region_props = load_dataframe(
                    context.working_location, props_key, converters={"KEY": str}
                )
                region_props.set_index(INDEX_COLUMNS, inplace=True)

                if props is None:
                    props = region_props
                    if region != "DEFAULT":
                        props = props.add_suffix(f".{region}")
                else:
                    props = props.join(region_props, on=INDEX_COLUMNS, rsuffix=f".{region}")

            all_coeffs.append(coeffs)
            all_props.append(props)

        results_data = pd.concat(all_results)
        coeffs_data = pd.concat(all_coeffs)
        props_data = None if any(prop is None for prop in all_props) else pd.concat(all_props)

        # Filter coefficient outliers.
        if parameters.outlier is not None:
            outlier_filter = abs(
                coeffs_data - coeffs_data.mean()
            ) <= parameters.outlier * coeffs_data.std(ddof=1)
            coeffs_data = coeffs_data[outlier_filter].dropna()

        # Join results, coefficients, and properties data.
        if props_data is None:
            data = coeffs_data.join(results_data, on=INDEX_COLUMNS).reset_index()
        else:
            data = coeffs_data.join(props_data, on=INDEX_COLUMNS)
            data = data.join(results_data, on=INDEX_COLUMNS).reset_index()

        # Filter for cell phase and selected ticks.
        data = data[data["PHASE"].isin(parameters.valid_phases)]
        data = data[data["TICK"].isin(parameters.valid_ticks)]

        # Convert units.
        convert_model_units(data, parameters.ds, parameters.dt, parameters.regions)

        # Remove nans.
        nan_indices = np.isnan(data.filter(like="shcoeffs")).any(axis=1)
        data = data[~nan_indices]
        nan_indices = np.isnan(data.filter(like="CENTER")).any(axis=1)
        data = data[~nan_indices]

        # Save final dataframe.
        save_dataframe(context.working_location, data_key, data, index=False)


@flow(name="analyze-cell-shapes_fit-model")
def run_flow_fit_model(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    """
    Analyze cell shapes subflow for fitting PCA model.

    Fit PCA for each key and save the resulting PCA object as a pickle. If the
    model already exits for a given key, that key is skipped.
    """

    shapes_path_key = make_key(series.name, "analysis", "analysis.SHAPES")
    pca_path_key = make_key(series.name, "analysis", "analysis.PCA")
    region_key = "_".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

    for key in keys:
        data_key = make_key(shapes_path_key, f"{series.name}_{key}_{region_key}.SHAPES.csv")
        model_key = make_key(pca_path_key, f"{series.name}_{key}_{region_key}.PCA.pkl")

        if check_key(context.working_location, model_key):
            continue

        data = load_dataframe.with_options(**OPTIONS)(context.working_location, data_key)

        coeffs = data.filter(like="shcoeffs").values
        ordering = data["NUM_VOXELS"].values
        model = fit_pca_model(coeffs, parameters.components, ordering)

        save_pickle(context.working_location, model_key, model)


@flow(name="analyze-cell-shapes_analyze-stats")
def run_flow_analyze_stats(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    """
    Analyze cell shapes subflow for analyzing distribution statistics.

    Perform statistical analysis of shape distributions. If the analysis file
    already exists for a given key, that key is skipped.
    """

    shapes_path_key = make_key(series.name, "analysis", "analysis.SHAPES")
    stats_path_key = make_key(series.name, "analysis", "analysis.STATISTICS")
    region_key = "_".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

    if parameters.reference is None:
        return

    ref_data = load_dataframe.with_options(**OPTIONS)(
        context.working_location, parameters.reference["data"]
    )
    ref_model = load_pickle.with_options(**OPTIONS)(
        context.working_location, parameters.reference["model"]
    )

    features = [
        f"{feature}.{region}" if region != "DEFAULT" else feature
        for region in parameters.regions
        for feature in ["volume", "height"]
    ]

    for key in keys:
        stats_key = make_key(stats_path_key, f"{series.name}_{key}_{region_key}.STATISTICS.csv")

        if check_key(context.working_location, stats_key):
            continue

        data_key = make_key(shapes_path_key, f"{series.name}_{key}_{region_key}.SHAPES.csv")
        data = load_dataframe.with_options(**OPTIONS)(context.working_location, data_key)

        all_stats = []

        for sample in range(parameters.sample_replicates):
            sample_data = (
                data.sample(frac=1, random_state=sample)
                .groupby("TICK")
                .head(parameters.sample_size)
            )

            feature_stats = calculate_feature_statistics(features, sample_data, ref_data)
            shape_stats = calculate_shape_statistics(
                ref_model, sample_data, ref_data, parameters.components
            )

            stats = pd.concat([feature_stats, shape_stats])
            stats["INDEX"] = sample

            all_stats.append(stats)

        all_stats_df = pd.concat(all_stats)

        save_dataframe(context.working_location, stats_key, all_stats_df, index=False)
