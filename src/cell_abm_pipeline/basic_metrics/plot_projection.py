import numpy as np
from matplotlib.patches import Rectangle

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

    def run(
        self,
        region=None,
        frames=(0, 1, 1),
        box=(100, 100, 10),
        ds=1,
        dt=1,
        scale=100,
        timestamp=True,
        scalebar=True,
    ):
        for seed in self.context.seeds:
            data = {}

            for key in self.context.keys:
                file = make_full_key(self.folders, self.files, "input", (key, seed))
                full_key = f"{self.context.name}_{key}_{seed:04d}"
                full_key = full_key.replace("__", "_")
                data[key] = (full_key, load_tar(self.context.working, file))

            annotations = {
                "timestamp": {
                    "include": timestamp,
                    "dt": dt,
                },
                "scalebar": {"include": scalebar, "ds": ds, "scale": scale},
            }

            self.plot_projection(data, seed, region, frames, box, annotations)

    def plot_projection(self, data, seed, region, frames, box, annotations):
        frame_keys = []

        for frame in np.arange(*frames):
            frame_group = {
                key: load_tar_member(tar, f"{prefix}_{frame:06d}.LOCATIONS.json")
                for key, (prefix, tar) in data.items()
            }

            annotations["timestamp"]["frame"] = frame

            make_plot(
                self.context.keys,
                frame_group,
                lambda a, d, k: self._plot_projection(a, d, k, region, box, annotations),
                size=5,
            )

            frame_key = make_full_key(self.folders, self.files, "frame", (seed, frame), region)
            save_plot(self.context.working, frame_key)
            frame_keys.append(frame_key)

        file_key = make_full_key(self.folders, self.files, "output", seed, region)
        save_gif(self.context.working, file_key, frame_keys)

    @staticmethod
    def _plot_projection(ax, data, key, region, box, annotations):
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        length, width, height = box
        array = np.zeros((length, width, height))
        borders = np.zeros((width, length))

        ax.set_xlim([0, length - 1])
        ax.set_ylim([width - 1, 0])

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

        PlotProjection.add_frame_timestamp(ax, length, width, **annotations["timestamp"])
        PlotProjection.add_frame_scalebar(ax, length, width, **annotations["scalebar"])

    @staticmethod
    def add_frame_timestamp(ax, length, width, include, dt, frame):
        if not include:
            return

        hours, minutes = divmod(frame * dt, 1)
        timestamp = f"{int(hours):02d}H:{int(minutes*60):02d}M"
        ax.text(
            0.03 * length,
            0.96 * width,
            timestamp,
            fontfamily="monospace",
            fontsize=10,
            color="w",
            fontweight="bold",
        )

    @staticmethod
    def add_frame_scalebar(ax, length, width, include, ds, scale):
        if not include:
            return

        scalebar = scale / ds
        ax.add_patch(
            Rectangle(
                (0.97 * length - scalebar, 0.92 * width),
                scalebar,
                0.01 * width,
                snap=True,
                color="w",
            )
        )
        ax.text(
            0.97 * length - scalebar / 2,
            0.97 * width,
            f"{scale} $\mu$m",
            fontsize=8,
            color="w",
            horizontalalignment="center",
        )
