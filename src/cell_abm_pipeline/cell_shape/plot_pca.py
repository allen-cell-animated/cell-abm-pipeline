import numpy as np

from cell_abm_pipeline.cell_shape.__config__ import CELL_FEATURES
from cell_abm_pipeline.cell_shape.calculate_coefficients import CalculateCoefficients
from cell_abm_pipeline.cell_shape.perform_pca import PerformPCA
from cell_abm_pipeline.utilities.load import load_pickle
from cell_abm_pipeline.utilities.save import save_plot
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key, make_full_key
from cell_abm_pipeline.utilities.plot import make_plot, make_legend


class PlotPCA:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "analysis", "SHPCA", True),
            "output": make_folder_key(context.name, "plots", "SHPCA", True),
        }
        self.files = {
            "input": lambda r: make_file_key(context.name, ["SHPCA", r, "pkl"], "%s", ""),
            "output": lambda r: make_file_key(context.name, ["SHPCA", r, "png"], "%s", ""),
        }

    def run(self, features=[], components=[], region=None, reference=None):
        data = {}

        for key in self.context.keys:
            key_file = make_full_key(self.folders, self.files, "input", key, region)
            data[key] = load_pickle(self.context.working, key_file)

        if reference:
            data["_reference"] = load_pickle(self.context.working, reference)

        self.plot_pca_variance_explained(data, region)

        for feature in features:
            self.plot_pca_transform_features(data, feature, region)

        if reference:
            for component in components:
                self.plot_pca_transform_compare(data, component, region)

    def plot_pca_variance_explained(self, data, region):
        make_plot(
            self.context.keys,
            data,
            self._plot_pca_variance_explained,
            xlabel="Component",
            ylabel="Explained variance (%)",
            legend=True,
        )

        plot_key = make_full_key(self.folders, self.files, "output", "variance_explained", region)
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_pca_variance_explained(ax, data, key):
        sim_var = np.cumsum(data[key]["pca"].explained_variance_ratio_)
        ax.plot(100 * sim_var, "-o", color="#555", markersize=3, label="data")

        if "_reference" in data:
            ref_var = np.cumsum(data["_reference"]["pca"].explained_variance_ratio_)
            ax.plot(100 * ref_var, "-o", color="#aaa", markersize=3, label="ref")

        ax.set_ylim([0, 100])
        ax.set_xticks(np.arange(0, len(sim_var), 1))
        ax.set_xticklabels(np.arange(1, len(sim_var) + 1, 1))

    def plot_pca_transform_features(self, data, feature, region):
        data["_feature"] = feature
        data["_region"] = region
        legend = make_legend(feature, CELL_FEATURES[feature])

        make_plot(
            self.context.keys,
            data,
            self._plot_pca_transform_features,
            xlabel="PC 1",
            ylabel="PC 2",
            legend={"handles": legend},
        )

        feature_key = f"transform_features_{feature}"
        plot_key = make_full_key(self.folders, self.files, "output", feature_key, region)
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_pca_transform_features(ax, data, key):
        df = data[key]["data"]
        pca = data[key]["pca"]
        region = data["_region"]
        feature = data["_feature"]

        bounds = CELL_FEATURES[feature]
        coeff_names = CalculateCoefficients.get_coeff_names()

        if region:
            feature = f"{feature}.{region}"
            coeff_names = coeff_names + CalculateCoefficients.get_coeff_names(suffix=f".{region}")

        transformed, nan_indices = PerformPCA.apply_data_transform(df[coeff_names], pca)

        ax.scatter(
            transformed[:, 0],
            transformed[:, 1],
            c=df[feature][~nan_indices],
            vmin=bounds[0],
            vmax=bounds[1],
            s=2,
            cmap="magma_r",
        )

    def plot_pca_transform_compare(self, data, component, region):
        pc = component + 1
        data["_component"] = component
        data["_region"] = region

        make_plot(
            self.context.keys,
            data,
            self._plot_pca_transform_compare,
            xlabel=f"PC {pc}",
            sharey="none",
        )

        component_key = f"transform_component_PC{pc}"
        plot_key = make_full_key(self.folders, self.files, "output", component_key, region)
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_pca_transform_compare(ax, data, key):
        df = data[key]["data"]
        df_ref = data["_reference"]["data"]
        pca = data["_reference"]["pca"]
        region = data["_region"]
        component = data["_component"]

        coeff_names = CalculateCoefficients.get_coeff_names()

        if region:
            coeff_names = coeff_names + CalculateCoefficients.get_coeff_names(suffix=f".{region}")

        reference, _ = PerformPCA.apply_data_transform(df_ref[coeff_names], pca)
        transformed, _ = PerformPCA.apply_data_transform(df[coeff_names], pca)

        ax.hist(reference[:, component], bins=20, density=True, alpha=0.3, color="black")
        ax.hist(transformed[:, component], bins=20, color="black", density=True, histtype="step")
