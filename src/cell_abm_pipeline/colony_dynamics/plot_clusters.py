import ast

from cell_abm_pipeline.utilities.load import load_dataframe
from cell_abm_pipeline.utilities.save import save_plot
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key
from cell_abm_pipeline.utilities.plot import make_plot


class PlotClusters:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "analysis", "CLUSTERS", True),
            "output": make_folder_key(context.name, "plots", "CLUSTERS", True),
        }
        self.files = {
            "input": make_file_key(context.name, ["CLUSTERS", "csv"], "%s", ""),
            "output": make_file_key(context.name, ["CLUSTERS", "png"], "%s", ""),
        }

    def run(self):
        data = {}

        for key in self.context.keys:
            key_file = self.folders["input"] + self.files["input"] % key
            data[key] = load_dataframe(self.context.working, key_file)

        self.plot_cluster_counts(data)
        self.plot_cluster_size_mean(data)
        self.plot_cluster_size_std(data)
        self.plot_cluster_fraction(data)
        self.plot_inter_cluster_distances_mean(data)
        self.plot_inter_cluster_distances_std(data)
        self.plot_intra_cluster_distances_mean(data)
        self.plot_intra_cluster_distances_std(data)

    def plot_cluster_counts(self, data):
        make_plot(
            self.context.keys,
            data,
            self._plot_cluster_counts,
            xlabel="Tick",
            ylabel="Number of Clusters",
        )

        plot_key = self.folders["output"] + self.files["output"] % "counts"
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_cluster_counts(ax, data, key):
        for _, seed in data[key].sort_values(by="TICK").groupby("SEED"):
            tick = seed["TICK"]
            clusters = seed["NUM_CLUSTERS"]
            singles = seed["NUM_SINGLES"]

            ax.plot(tick, clusters, c="#f00", alpha=0.5)
            ax.plot(tick, singles, c="#00f", alpha=0.5)

    def plot_cluster_size_mean(self, data):
        make_plot(
            self.context.keys,
            data,
            self._plot_cluster_size_mean,
            xlabel="Tick",
            ylabel="Cluster Size Mean",
        )

        plot_key = self.folders["output"] + self.files["output"] % "size_mean"
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_cluster_size_mean(ax, data, key):
        for _, seed in data[key].sort_values(by="TICK").groupby("SEED"):
            tick = seed["TICK"]
            mean = seed["CLUSTER_SIZE_MEAN"]
            ax.plot(tick, mean, c="#000", alpha=0.5)

    def plot_cluster_size_std(self, data):
        make_plot(
            self.context.keys,
            data,
            self._plot_cluster_size_std,
            xlabel="Tick",
            ylabel="Cluster Size Std Dev",
        )

        plot_key = self.folders["output"] + self.files["output"] % "size_std"
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_cluster_size_std(ax, data, key):
        for _, seed in data[key].sort_values(by="TICK").groupby("SEED"):
            tick = seed["TICK"]
            mean = seed["CLUSTER_SIZE_STD"]
            ax.plot(tick, mean, c="#000", alpha=0.5)

    def plot_cluster_fraction(self, data):
        make_plot(
            self.context.keys,
            data,
            self._plot_cluster_fraction,
            xlabel="Tick",
            ylabel="Fraction in Clusters",
        )

        plot_key = self.folders["output"] + self.files["output"] % "fraction"
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_cluster_fraction(ax, data, key):
        for _, seed in data[key].sort_values(by="TICK").groupby("SEED"):
            tick = seed["TICK"]
            clusters = seed["CLUSTER_SIZE_TOTAL"]
            singles = seed["NUM_SINGLES"]
            ax.plot(tick, clusters / (clusters + singles), c="#000", alpha=0.5)

    def plot_inter_cluster_distances_mean(self, data):
        make_plot(
            self.context.keys,
            data,
            self._plot_inter_cluster_distances_mean,
            xlabel="Tick",
            ylabel="Intercluster Distance Mean",
        )

        plot_key = self.folders["output"] + self.files["output"] % "inter_distances_mean"
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_inter_cluster_distances_mean(ax, data, key):
        for _, seed in data[key].sort_values(by="TICK").groupby("SEED"):
            tick = seed["TICK"]
            mean = seed["INTER_DISTANCE_MEAN"]
            ax.plot(tick, mean, c="#000", alpha=0.5)

    def plot_inter_cluster_distances_std(self, data):
        make_plot(
            self.context.keys,
            data,
            self._plot_inter_cluster_distances_std,
            xlabel="Tick",
            ylabel="Intercluster Distance Std Dev",
        )

        plot_key = self.folders["output"] + self.files["output"] % "inter_distances_std"
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_inter_cluster_distances_std(ax, data, key):
        for _, seed in data[key].sort_values(by="TICK").groupby("SEED"):
            tick = seed["TICK"]
            mean = seed["INTER_DISTANCE_STD"]
            ax.plot(tick, mean, c="#000", alpha=0.5)

    def plot_intra_cluster_distances_mean(self, data):
        make_plot(
            self.context.keys,
            data,
            self._plot_intra_cluster_distances_mean,
            xlabel="Tick",
            ylabel="Intracluster Distance Mean",
        )

        plot_key = self.folders["output"] + self.files["output"] % "intra_distances_mean"
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_intra_cluster_distances_mean(ax, data, key):
        for _, seed in data[key].sort_values(by="TICK").groupby("SEED"):
            ticks = seed["TICK"]
            means = seed["INTRA_DISTANCE_MEAN"]
            flattened = [
                (tick, m)
                for tick, mean in zip(ticks, means)
                for m in ast.literal_eval(mean)
                if mean
            ]

            if len(flattened) > 0:
                ticks, means = zip(*flattened)
                ax.scatter(ticks, means, s=5, c="#000", alpha=0.5, edgecolors="none")

    def plot_intra_cluster_distances_std(self, data):
        make_plot(
            self.context.keys,
            data,
            self._plot_intra_cluster_distances_std,
            xlabel="Tick",
            ylabel="Intracluster Distance Std Dev",
        )

        plot_key = self.folders["output"] + self.files["output"] % "intra_distances_std"
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_intra_cluster_distances_std(ax, data, key):
        for _, seed in data[key].sort_values(by="TICK").groupby("SEED"):
            ticks = seed["TICK"]
            stds = seed["INTRA_DISTANCE_STD"]
            flattened = [
                (tick, s) for tick, std in zip(ticks, stds) for s in ast.literal_eval(std) if std
            ]

            if len(flattened) > 0:
                ticks, stds = zip(*flattened)
                ax.scatter(ticks, stds, s=5, c="#000", alpha=0.5, edgecolors="none")
