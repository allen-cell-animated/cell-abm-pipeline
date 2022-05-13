import pandas as pd
from matplotlib.lines import Line2D

from cell_abm_pipeline.basic_metrics.__config__ import PHASE_COLORS
from cell_abm_pipeline.utilities.load import load_dataframe
from cell_abm_pipeline.utilities.save import save_plot
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key, make_full_key
from cell_abm_pipeline.utilities.plot import make_plot, make_legend


class PlotSpatial:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "results", "", False),
            "output": make_folder_key(context.name, "plots", "BASIC", True),
        }
        self.files = {
            "input": make_file_key(context.name, ["csv"], "%s", "%04d"),
            "output": lambda r: make_file_key(context.name, ["BASIC", r, "png"], "%s", "%04d"),
        }

    def run(self, ds=1, region=None):
        for seed in self.context.seeds:
            data = {}

            for key in self.context.keys:
                file = make_full_key(self.folders, self.files, "input", (key, seed))
                file_data = load_dataframe(self.context.working, file)
                self.convert_data_units(file_data, ds, region)
                data[key] = file_data

            self.plot_volume_distribution(data, seed, region)

            if not region:
                self.plot_phase_distribution(data, seed)
                self.plot_population_distribution(data, seed)

    @staticmethod
    def convert_data_units(data, ds, region=None):
        data["VOLUME"] = ds * ds * ds * data["NUM_VOXELS"]

        if region:
            data[f"VOLUME.{region}"] = ds * ds * ds * data[f"NUM_VOXELS.{region}"]

    def plot_volume_distribution(self, data, seed, region=None, vmin=0, vmax=4000):
        legend = make_legend("VOLUME", [vmin, vmax])

        make_plot(
            self.context.keys,
            data,
            lambda a, d, k: self._plot_volume_distribution(a, d, k, region, vmin, vmax),
            legend={"handles": legend},
        )

        plot_key = make_full_key(self.folders, self.files, "output", ("volumes", seed), region)
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_volume_distribution(ax, data, key, region, vmin, vmax):
        value = f"VOLUME.{region}" if region else "VOLUME"
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        data = data[key]
        data = data[data["TICK"] == data["TICK"].max()]

        x = data["CENTER_X"]
        y = data["CENTER_Y"]
        v = data[value]

        ax.scatter(x, y, c=v, s=10, cmap="magma_r", vmin=vmin, vmax=vmax)

    def plot_phase_distribution(self, data, seed):
        legend = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=phase,
                markerfacecolor=color,
                markersize=5,
            )
            for phase, color in PHASE_COLORS.items()
        ]

        make_plot(
            self.context.keys,
            data,
            self._plot_phase_distribution,
            legend={"handles": legend},
        )

        plot_key = make_full_key(self.folders, self.files, "output", ("phases", seed))
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_phase_distribution(ax, data, key):
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        data = data[key]
        data = data[data["TICK"] == data["TICK"].max()]

        x = data["CENTER_X"]
        y = data["CENTER_Y"]
        phases = [PHASE_COLORS[phase] for phase in data["PHASE"]]

        ax.scatter(x, y, c=phases, s=10)

    def plot_population_distribution(self, data, seed):
        make_plot(
            self.context.keys,
            data,
            self._plot_population_distribution,
        )

        plot_key = make_full_key(self.folders, self.files, "output", ("populations", seed))
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_population_distribution(ax, data, key):
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        data = data[key]
        data = data[data["TICK"] == data["TICK"].max()]

        x = data["CENTER_X"]
        y = data["CENTER_Y"]
        v = data["POPULATION"]

        ax.scatter(x, y, c=v, s=10, cmap="tab10", vmin=1, vmax=11)
