import ast
import numpy as np
from matplotlib import cm
from tqdm import tqdm

from cell_abm_pipeline.utilities.load import load_dataframe
from cell_abm_pipeline.utilities.save import save_plot, save_gif
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key
from cell_abm_pipeline.utilities.plot import make_plot


class PlotNeighbors:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "analysis", "NEIGHBORS", False),
            "output": make_folder_key(context.name, "plots", "NEIGHBORS", True),
        }
        self.files = {
            "input": make_file_key(context.name, ["NEIGHBORS", "csv", "xz"], "", "%04d"),
            "output_frame": make_file_key(context.name, ["NEIGHBORS", "%06d", "png"], "", "%04d"),
            "output": make_file_key(context.name, ["NEIGHBORS", "gif"], "", "%04d"),
        }

    def run(self):
        for seed in self.context.seeds:
            key_file = self.folders["input"] + self.files["input"] % seed
            data = load_dataframe(self.context.working, key_file)
            data = data[data.KEY.isin(self.context.keys)]
            self.plot_neighbors(data)

    def plot_neighbors(self, data):
        xlim = [data.CX.max(), data.CX.min()]
        ylim = [data.CY.max(), data.CY.min()]
        vmax = data.GROUP.max()
        cmap = cm.get_cmap("gist_rainbow", vmax)
        seed = data.SEED.unique()[0]

        frame_keys = []

        for tick, tick_group in tqdm(data.sort_values(by="TICK").groupby("TICK")):
            make_plot(
                self.context.keys,
                {"data": tick_group, "xlim": xlim, "ylim": ylim, "cmap": cmap},
                self._plot_neighbors,
            )

            frame_key = self.folders["output"] + self.files["output_frame"] % (seed, tick)
            save_plot(self.context.working, frame_key)
            frame_keys.append(frame_key)

        file_key = self.folders["output"] + self.files["output"] % seed
        save_gif(self.context.working, file_key, frame_keys)

    @staticmethod
    def _plot_neighbors(ax, data, key):
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        frame = data["data"][data["data"].KEY == key]
        x_centroids = {id: x for id, x in zip(frame.ID, frame.CX)}
        y_centroids = {id: y for id, y in zip(frame.ID, frame.CY)}

        connections = zip(frame.ID, frame.NEIGHBOR)
        for from_id, to_id in connections:
            if to_id == 0:
                continue

            ax.plot(
                [x_centroids[from_id], x_centroids[to_id]],
                [y_centroids[from_id], y_centroids[to_id]],
                color="k",
                lw=0.5,
                zorder=1,
            )

        ax.scatter(frame.CX, frame.CY, c=frame.GROUP, s=5, cmap=data["cmap"], zorder=2)

        ax.set_xlim(data["xlim"])
        ax.set_ylim(data["ylim"])
