import ast
import numpy as np

from cell_abm_pipeline.utilities.load import load_dataframe
from cell_abm_pipeline.utilities.save import save_plot
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key
from cell_abm_pipeline.utilities.plot import make_plot


class PlotMeasures:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "analysis", "MEASURES", True),
            "output": make_folder_key(context.name, "plots", "MEASURES", True),
        }
        self.files = {
            "input": make_file_key(context.name, ["MEASURES", "csv"], "%s", ""),
            "output": make_file_key(context.name, ["MEASURES", "png"], "%s", ""),
        }

    def run(self):
        data = {}

        for key in self.context.keys:
            key_file = self.folders["input"] + self.files["input"] % key
            data[key] = load_dataframe(self.context.working, key_file)

        self.plot_degree_distribution(data)
        self.plot_average_degree_mean(data)
        self.plot_average_degree_std(data)
        self.plot_network_distances(data)
        self.plot_network_centrality(data)

    def plot_degree_distribution(self, data):
        make_plot(
            self.context.keys,
            data,
            self._plot_degree_distribution,
            xlabel="Degree",
            ylabel="Frequency",
        )

        plot_key = self.folders["output"] + self.files["output"] % "degree_distribution"
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_degree_distribution(ax, data, key):
        final_tick = data[key].TICK.max()
        degree_data = data[key][data[key].TICK == final_tick]

        degrees = [deg for degree in degree_data["DEGREES"] for deg in ast.literal_eval(degree)]
        ax.bar(*np.unique(degrees, return_counts=True))

    def plot_average_degree_mean(self, data):
        make_plot(
            self.context.keys,
            data,
            self._plot_average_degree_mean,
            xlabel="Tick",
            ylabel="Average Degree Mean",
        )

        plot_key = self.folders["output"] + self.files["output"] % "average_degree_mean"
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_average_degree_mean(ax, data, key):
        for _, seed in data[key].sort_values(by="TICK").groupby("SEED"):
            tick = seed["TICK"]
            mean = seed["DEGREE_MEAN"]
            ax.plot(tick, mean, c="#000", alpha=0.5)

    def plot_average_degree_std(self, data):
        make_plot(
            self.context.keys,
            data,
            self._plot_average_degree_std,
            xlabel="Tick",
            ylabel="Cluster Size Std Dev",
        )

        plot_key = self.folders["output"] + self.files["output"] % "average_degree_std"
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_average_degree_std(ax, data, key):
        for _, seed in data[key].sort_values(by="TICK").groupby("SEED"):
            tick = seed["TICK"]
            mean = seed["DEGREE_STD"]
            ax.plot(tick, mean, c="#000", alpha=0.5)

    def plot_network_distances(self, data):
        make_plot(
            self.context.keys,
            data,
            self._plot_network_distances,
            xlabel="Tick",
            ylabel="Distance",
            legend=True,
        )

        plot_key = self.folders["output"] + self.files["output"] % "distances"
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_network_distances(ax, data, key):
        for _, seed in data[key].sort_values(by="TICK").groupby("SEED"):
            tick = seed["TICK"]
            radius = seed["RADIUS"]
            diameter = seed["DIAMETER"]
            eccentricity = seed["ECCENTRICITY"]
            shortest_path = seed["SHORTEST_PATH"]

            ax.plot(tick, eccentricity, c="#009", alpha=0.5, label="eccentricity")
            ax.plot(tick, shortest_path, c="#900", alpha=0.5, label="shortest path")
            ax.plot(tick, radius, c="#000", alpha=0.5, label="radius")
            ax.plot(tick, diameter, c="#090", alpha=0.5, label="diameter")

    def plot_network_centrality(self, data):
        make_plot(
            self.context.keys,
            data,
            self._plot_network_centrality,
            xlabel="Tick",
            ylabel="Centrality",
            legend=True,
        )

        plot_key = self.folders["output"] + self.files["output"] % "centrality"
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_network_centrality(ax, data, key):
        for _, seed in data[key].sort_values(by="TICK").groupby("SEED"):
            tick = seed["TICK"]
            degree = seed["DEGREE_CENTRALITY"]
            closeness = seed["CLOSENESS_CENTRALITY"]
            betweenness = seed["BETWEENNESS_CENTRALITY"]

            ax.plot(tick, degree, c="#000", alpha=0.5, label="degree")
            ax.plot(tick, closeness, c="#900", alpha=0.5, label="closeness")
            ax.plot(tick, betweenness, c="#009", alpha=0.5, label="betweenness")
