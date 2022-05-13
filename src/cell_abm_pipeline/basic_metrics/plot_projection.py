import numpy as np

from cell_abm_pipeline.utilities.load import load_tar, load_tar_member
from cell_abm_pipeline.utilities.save import save_plot, save_gif
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key, make_full_key
from cell_abm_pipeline.utilities.plot import make_plot


class PlotProjection:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "data", "LOCATIONS", False),
            "output": make_folder_key(context.name, "plots", "BASIC", True),
            "frame": make_folder_key(context.name, "plots", "BASIC", True),
        }
        self.files = {
            "input": make_file_key(context.name, ["LOCATIONS", "tar", "xz"], "%s", "%04d"),
            "output": lambda r: make_file_key(context.name, ["BASIC", r, "gif"], "", "%04d"),
            "frame": lambda r: make_file_key(context.name, ["BASIC", "%06d", r, "png"], "", "%04d"),
        }

    def run(self, region=None, frames=(0, 1, 1), box=(100, 100, 10)):
        for seed in self.context.seeds:
            data = {}

            for key in self.context.keys:
                file = make_full_key(self.folders, self.files, "input", (key, seed))
                full_key = f"{self.context.name}_{key}_{seed:04d}"
                full_key = full_key.replace("__", "_")
                data[key] = (full_key, load_tar(self.context.working, file))

            self.plot_projection(data, seed, region, frames, box)

    def plot_projection(self, data, seed, region, frames, box):
        frame_keys = []

        for frame in np.arange(*frames):
            frame_group = {
                key: load_tar_member(tar, f"{prefix}_{frame:06d}.LOCATIONS.json")
                for key, (prefix, tar) in data.items()
            }

            make_plot(
                self.context.keys,
                frame_group,
                lambda a, d, k: self._plot_projection(a, d, k, region, box),
                size=5,
            )

            frame_key = make_full_key(self.folders, self.files, "frame", (seed, frame), region)
            save_plot(self.context.working, frame_key)
            frame_keys.append(frame_key)

        file_key = make_full_key(self.folders, self.files, "output", seed, region)
        save_gif(self.context.working, file_key, frame_keys)

    @staticmethod
    def _plot_projection(ax, data, key, region, box):
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        length, width, height = box
        array = np.zeros((length, width, height))
        borders = np.zeros((length, width))

        for cell in data[key]:
            if region:
                all_voxels = [
                    voxels
                    for reg in cell["location"]
                    for voxels in reg["voxels"]
                    if reg["region"] == region
                ]
            else:
                all_voxels = [voxels for reg in cell["location"] for voxels in reg["voxels"]]

            array[tuple(np.transpose(all_voxels))] = cell["id"]

        for i in range(length):
            for j in range(width):
                for k in range(height):
                    target = array[i][j][k]
                    if target != 0:
                        neighbors = [
                            1
                            for ii in [-1, 0, 1]
                            for jj in [-1, 0, 1]
                            if array[i + ii][j + jj][k] == target
                        ]
                        borders[j][i] += 9 - sum(neighbors)

        normalize = borders.max()
        borders = borders / normalize

        ax.imshow(borders, cmap="bone", interpolation="none")
