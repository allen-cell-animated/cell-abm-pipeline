"""
Workflow for format figure panel data.
"""

from dataclasses import dataclass, field

from io_collection.keys import make_key
from io_collection.load import load_dataframe, load_pickle, load_tar
from io_collection.save import save_dataframe
from prefect import flow

PANELS: list[str] = [
    "feature_bins",
    "key_violins",
]

import pandas as pd
from arcade_collection.output import extract_feature_bins

from cell_abm_pipeline.flows.analyze_shape_modes import PCA_COMPONENTS


@dataclass
class ParametersConfig:
    """Parameter configuration for format panel data flow."""

    reference: dict

    box: tuple[int, int, int] = (1, 1, 1)

    frame: int = 0

    scale: float = 1.0

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])
    """List of subcellular regions."""

    components: int = PCA_COMPONENTS
    """Number of principal components (i.e. shape modes)."""

    panels: list[str] = field(default_factory=lambda: PANELS)


@dataclass
class ContextConfig:
    """Context configuration for format panel data flow."""

    working_location: str


@dataclass
class SeriesConfig:
    """Series configuration for format panel data flow."""

    name: str

    seeds: list[int]

    conditions: list[dict]


@flow(name="format-panel-data")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    """Main format panel data flow."""

    if "feature_bins" in parameters.panels:
        run_flow_format_feature_bins(context, series, parameters)

    if "key_violins" in parameters.panels:
        run_flow_format_key_violins(context, series, parameters)


@flow(name="format-panel-data_format-feature-bins")
def run_flow_format_feature_bins(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    data_key = make_key(series.name, "data", "data.LOCATIONS")
    panel_key = make_key(series.name, "panels")
    keys = [condition["key"] for condition in series.conditions]

    all_feature_bins = []

    for key in keys:
        tars = {}

        for seed in series.seeds:
            series_key = f"{series.name}_{key}_{seed:04d}"
            tar_key = make_key(data_key, f"{series_key}.LOCATIONS.tar.xz")
            tar = load_tar(context.working_location, tar_key)
            tars[series_key] = tar

        feature_bins = extract_feature_bins(tars, parameters.frame, parameters.scale)
        feature_bins["key"] = key
        all_feature_bins.append(feature_bins)

    feature_bins_key = make_key(panel_key, f"{series.name}.feature_bins.csv")
    save_dataframe(
        context.working_location, feature_bins_key, pd.concat(all_feature_bins), index=False
    )


@flow(name="format-panel-data_format-key-violins")
def run_flow_format_key_violins(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    analysis_key = make_key(series.name, "analysis", "analysis.PCA")
    panel_key = make_key(series.name, "panels")
    region_key = ":".join(sorted(parameters.regions))
    keys = [condition["key"] for condition in series.conditions]

    features = ["KEY", "ID", "SEED", "TICK"]
    features = features + [
        f"{feature}.{region}" if region != "DEFAULT" else feature
        for feature in ["volume", "height"]
        for region in parameters.regions
    ]
    modes = [f"PC{component + 1}" for component in range(parameters.components)]
    key_columns = features + modes

    key_violins = []

    ref_model = load_pickle(context.working_location, parameters.reference["model"])
    ref_data = load_dataframe(context.working_location, parameters.reference["data"])

    columns = ref_data.filter(like="shcoeffs").columns

    ref_transform = ref_model.transform(ref_data[columns].values)
    for component in range(parameters.components):
        ref_data[f"PC{component + 1}"] = ref_transform[:, component]

    ref_data.drop(columns=[col for col in ref_data if col not in key_columns], inplace=True)

    ref_means = ref_data.mean()
    ref_means["KEY"] = "reference_mean"
    key_violins.append(ref_means.to_frame().T)

    ref_stds = ref_data.std(ddof=1)
    ref_stds["KEY"] = "reference_std"
    key_violins.append(ref_stds.to_frame().T)

    for key in keys:
        pca_data_key = make_key(analysis_key, f"{series.name}_{key}_{region_key}.PCA.csv")
        pca_data = load_dataframe(context.working_location, pca_data_key)

        key_transform = ref_model.transform(pca_data[columns].values)
        for component in range(parameters.components):
            pca_data[f"PC{component + 1}"] = key_transform[:, component]

        pca_data.drop(columns=[col for col in pca_data if col not in key_columns], inplace=True)
        key_violins.append(pca_data)

    key_violins_key = make_key(panel_key, f"{series.name}.key_violins.csv")
    save_dataframe(context.working_location, key_violins_key, pd.concat(key_violins), index=False)
