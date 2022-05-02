import pandas as pd
from matplotlib.lines import Line2D

from cell_abm_pipeline.basic_metrics.__config__ import PHASE_COLORS
from cell_abm_pipeline.utilities.load import load_dataframe
from cell_abm_pipeline.utilities.save import save_plot
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key
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
            "output": make_file_key(context.name, ["BASIC", "png"], "%s", "%04d"),
        }

    def run(self):
        for seed in self.context.seeds:
            data = {}

            for key in self.context.keys:
                file = self.folders["input"] + self.files["input"] % (key, seed)
                data[key] = load_dataframe(self.context.working, file)

            # self.plot_volume_distribution(data, seed)
            self.plot_phase_distribution(data, seed)

    def plot_volume_distribution(self, data, seed, vmin=500, vmax=2000):
        legend = make_legend("NUM_VOXELS", [vmin, vmax])

        make_plot(
            self.context.keys,
            data,
            lambda a, d, k: self._plot_volume_distribution(a, d, k, vmin, vmax),
            legend={"handles": legend},
        )

        plot_key = self.folders["output"] + self.files["output"] % ("volume_distribution", seed)
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_volume_distribution(ax, data, key, vmin, vmax):
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        data = data[key]
        data = data[data.TICK == data.TICK.max()]

        x = data.CENTER_X
        y = data.CENTER_Y
        v = data.NUM_VOXELS

        ax.scatter(x, y, c=v, s=5, cmap="magma_r", vmin=vmin, vmax=vmax)

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

        plot_key = self.folders["output"] + self.files["output"] % ("phase_distribution", seed)
        save_plot(self.context.working, plot_key)

    @staticmethod
    def _plot_phase_distribution(ax, data, key):
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        data = data[key]
        data = data[data.TICK == data.TICK.max()]

        x = data.CENTER_X
        y = data.CENTER_Y
        phases = [PHASE_COLORS[phase] for phase in data["PHASE"]]

        ax.scatter(x, y, c=phases, s=10)
