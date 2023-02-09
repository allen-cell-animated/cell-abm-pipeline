from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from abm_shape_collection import calculate_shape_stats, calculate_size_stats, fit_pca_model
from arcade_collection.output import convert_model_units
from io_collection.keys import check_key, make_key
from io_collection.load import load_dataframe, load_pickle
from io_collection.save import save_dataframe, save_pickle
from prefect import flow

PCA_COMPONENTS = 8

INDEX_COLUMNS = ["KEY", "ID", "SEED", "TICK"]

VALID_PHASES = ["PROLIFERATIVE_G1", "PROLIFERATIVE_S", "PROLIFERATIVE_G2"]


@dataclass
class ParametersConfig:
    reference: Optional[dict] = None

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])

    components: int = PCA_COMPONENTS
    """Number of principal components (i.e. shape modes)."""

    valid_phases: list[str] = field(default_factory=lambda: VALID_PHASES)
    """Valid phases for calculating shape modes."""

    ds: float = 1.0

    dt: float = 1.0


@dataclass
class ContextConfig:
    working_location: str


@dataclass
class SeriesConfig:
    name: str

    seeds: list[int]

    conditions: list[dict]


@flow(name="run-shape-analysis")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    # Process spherical harmonics coefficients and parsed simulation results and
    # compile into a single dataframe that can used for PCA. If the combined
    # data already exists for a given key, that key is skipped.
    run_flow_process_data(context, series, parameters)

    # Fit PCA for each key and save the resulting PCA object as a pickle. If the
    # model already exits for a given key, that key is skipped.
    run_flow_fit_model(context, series, parameters)

    # Perform statistical analysis of shape distributions.
    run_flow_calculate_stats(context, series, parameters)


@flow(name="run-shape-analysis_load-data")
def run_flow_process_data(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    path_key = make_key(series.name, "analysis", "analysis.PCA")
    region_key = ":".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

    for key in keys:
        data_key = make_key(path_key, f"{series.name}_{key}_{region_key}.PCA.csv")

        if check_key(context.working_location, data_key):
            continue

        all_coeffs = []
        all_results = []

        for seed in series.seeds:
            coeffs = None

            # Load coefficients for each region
            for region in parameters.regions:
                coeffs_key = make_key(
                    series.name,
                    "analysis",
                    "analysis.COEFFS",
                    f"{series.name}_{key}_{seed:04d}_{region}.COEFFS.csv",
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

            all_coeffs.append(coeffs)

            # Load parsed results
            results_key = make_key(series.name, "results", f"{series.name}_{key}_{seed:04d}.csv")
            results = load_dataframe(context.working_location, results_key)
            results["KEY"] = key
            results["SEED"] = seed
            results.set_index(INDEX_COLUMNS, inplace=True)
            all_results.append(results)

        coeffs_data = pd.concat(all_coeffs)
        results_data = pd.concat(all_results)

        data = coeffs_data.join(results_data, on=INDEX_COLUMNS).reset_index()
        data = data[data["PHASE"].isin(parameters.valid_phases)]
        convert_model_units(data, parameters.ds, parameters.dt, parameters.regions)

        # Remove nans
        nan_indices = np.isnan(data.filter(like="shcoeffs")).any(axis=1)
        data = data[~nan_indices]

        save_dataframe(context.working_location, data_key, data, index=False)


@flow(name="run-shape-analysis_fit-model")
def run_flow_fit_model(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    path_key = make_key(series.name, "analysis", "analysis.PCA")
    region_key = ":".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

    for key in keys:
        data_key = make_key(path_key, f"{series.name}_{key}_{region_key}.PCA.csv")
        model_key = make_key(path_key, f"{series.name}_{key}_{region_key}.PCA.pkl")

        if check_key(context.working_location, model_key):
            continue

        data = load_dataframe(context.working_location, data_key)
        data = data[data["SEED"].isin(series.seeds)]

        coeffs = data.filter(like="shcoeffs").values
        ordering = data["NUM_VOXELS"].values
        model = fit_pca_model(coeffs, parameters.components, ordering)

        save_pickle(context.working_location, model_key, model)


@flow(name="run-shape-analysis_calculate-stats")
def run_flow_calculate_stats(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    path_key = make_key(series.name, "analysis", "analysis.PCA")
    region_key = ":".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

    if parameters.reference is None:
        return

    ref_data = load_dataframe(context.working_location, parameters.reference["data"])
    ref_model = load_pickle(context.working_location, parameters.reference["model"])

    for key in keys:
        data_key = make_key(path_key, f"{series.name}_{key}_{region_key}.PCA.csv")
        data = load_dataframe(context.working_location, data_key)
        data = data[data["SEED"].isin(series.seeds)]

        size_stats = calculate_size_stats(data, ref_data, parameters.regions)
        shape_stats = calculate_shape_stats(ref_model, data, ref_data, parameters.components)
        stats = pd.concat([size_stats, shape_stats])

        stats_key = make_key(
            series.name,
            "analysis",
            "analysis.STATS",
            f"{series.name}_{key}_{region_key}.STATS.csv",
        )
        save_dataframe(context.working_location, stats_key, stats, index=False)
