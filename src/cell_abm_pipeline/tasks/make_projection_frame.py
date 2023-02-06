import numpy as np
from arcade_collection.output import extract_tick_json, get_location_voxels
from matplotlib.patches import Rectangle
from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def make_projection_frame(keys, data, name, tick, scale, ds, dt, box, region=None):
    fig, gridspec, indices = make_grid_figure(keys)
    length, width, height = box

    for i, j, (key, seed) in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(f"{key} [{seed}]")
        ax.invert_yaxis()
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_xlim([0, length - 1])
        ax.set_ylim([width - 1, 0])

        tick_json = extract_tick_json.fn(
            data[(key, seed)], f"{name}_{key}_{seed:04d}", tick, "LOCATIONS"
        )
        array = create_projection_array(tick_json, length, width, height, region)

        ax.imshow(array, cmap="bone", interpolation="none", vmin=0, vmax=1)

        add_frame_timestamp(ax, length, width, dt, tick)
        add_frame_scalebar(ax, length, width, ds, scale)

    return fig


def create_projection_array(data, length, width, height, region=None):
    array = np.zeros((length, width, height))
    borders = np.zeros((width, length))

    for cell in data:
        voxels = get_location_voxels.fn(cell, region)
        array[tuple(np.transpose(voxels))] = cell["id"]

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

    return borders


def add_frame_timestamp(ax, length, width, dt, frame):
    hours, minutes = divmod(round(frame * dt), 1)
    timestamp = f"{int(hours):02d}H:{int(minutes*60):02d}M"

    ax.text(
        0.03 * length,
        0.96 * width,
        timestamp,
        fontfamily="monospace",
        fontsize=20,
        color="w",
        fontweight="bold",
    )


def add_frame_scalebar(ax, length, width, ds, scale):
    scalebar = scale / ds

    ax.add_patch(
        Rectangle(
            (0.95 * length - scalebar, 0.92 * width),
            scalebar,
            0.01 * width,
            snap=True,
            color="w",
        )
    )

    ax.text(
        0.95 * length - scalebar / 2,
        0.975 * width,
        f"{scale} $\mu$m",
        fontsize=10,
        color="w",
        horizontalalignment="center",
    )
