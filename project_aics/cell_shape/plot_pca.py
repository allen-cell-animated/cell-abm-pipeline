import numpy as np

from project_aics.cell_shape.__config__ import CELL_FEATURES
from project_aics.cell_shape.calculate_coefficients import CalculateCoefficients
from project_aics.utilities.load import load_pickle_from_fs
from project_aics.utilities.save import save_plot
from project_aics.utilities.keys import make_folder_key, make_file_key
from project_aics.utilities.plot import make_plot, make_legend


class PlotPCA:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "analysis", "SHPCA", True),
            "output": make_folder_key(context.name, "plots", "SHPCA", True),
        }
        self.files = {
            "input": lambda r: make_file_key(context.name, ["SHPCA", r, "pkl"], "%s", ""),
            "output": make_file_key(context.name, ["SHPCA", "png"], "%s", ""),
        }

    def run(self, features=[], region=None):
        data = {}

        for key in self.context.keys:
            key_file = self.folders["input"] + self.files["input"](region) % key
            data[key] = load_pickle_from_fs(self.context.working, key_file)

        self.plot_pca_variance_explained(data)

        for feature in features:
            self.plot_pca_transform_features(data, feature)

    def plot_pca_variance_explained(self, data):
        make_plot(
            self.context.keys,
            data,
            self._plot_pca_variance_explained,
            xlabel="Component",
            ylabel="Explained variance (%)",
            legend=True,
        )

        plot_key = self.folders["output"] + self.files["output"] % "variance_explained"
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_pca_variance_explained(ax, data, key):
        sim_var = np.cumsum(data[key]["pca"].explained_variance_ratio_)

        ax.plot(100 * sim_var, "-o", color="#555", markersize=3, label="sim")
        ax.set_ylim([0, 100])
        ax.set_xticks(np.arange(0, len(sim_var), 1))
        ax.set_xticklabels(np.arange(1, len(sim_var) + 1, 1))

    def plot_pca_transform_features(self, data, feature):
        data["_feature"] = feature
        legend = make_legend(feature, CELL_FEATURES[feature])

        make_plot(
            self.context.keys,
            data,
            self._plot_pca_transform_features,
            xlabel="PC 1",
            ylabel="PC 2",
            legend={"handles": legend},
        )

        plot_key = self.folders["output"] + self.files["output"] % f"transform_features_{feature}"
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_pca_transform_features(ax, data, key):
        df = data[key]["data"]
        pca = data[key]["pca"]
        feature = data["_feature"]

        bounds = CELL_FEATURES[feature]
        coeff_names = CalculateCoefficients.get_coeff_names()
        transformed = pca.transform(df[coeff_names].values)

        ax.scatter(
            transformed[:, 0],
            transformed[:, 1],
            c=df[feature],
            vmin=bounds[0],
            vmax=bounds[1],
            s=2,
            cmap="magma_r",
        )
