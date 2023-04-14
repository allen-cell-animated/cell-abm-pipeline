from math import ceil, sqrt
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PLOT_SIZE = 4

mpl.use("Agg")
mpl.rc("figure", dpi=200)
mpl.rc("font", size=8)
mpl.rc("legend", fontsize=6)
mpl.rc("axes", titlesize=10, titleweight="bold")


def make_single_figure() -> mpl.figure.Figure:
    fig = plt.figure(constrained_layout=True)
    return fig


def make_grid_figure(keys: list[Any], size: int = PLOT_SIZE) -> mpl.figure.Figure:
    n_rows, n_cols, indices = separate_rows_cols(keys)
    fig = plt.figure(figsize=(n_cols * size, n_rows * size), constrained_layout=True)
    gridspec = fig.add_gridspec(n_rows, n_cols)
    return fig, gridspec, indices


def separate_rows_cols(keys: list[Any]) -> tuple[int, int, list[tuple[int, int, Any]]]:
    if isinstance(keys[0], tuple):
        cols = list(dict.fromkeys([a for a, b in keys]))
        rows = list(dict.fromkeys([b for a, b in keys]))
        n_cols = len(cols)
        n_rows = len(rows)

        all_tuple_indices = [
            (i, j, (col, row)) for i, row in enumerate(rows) for j, col in enumerate(cols)
        ]
        indices = [(i, j, key) for i, j, key in all_tuple_indices if key in keys]
    else:
        n_items = len(keys)
        n_cols = ceil(sqrt(n_items))
        n_rows = ceil(n_items / n_cols)

        all_key_indices = [(i, j, i * n_cols + j) for i in range(n_rows) for j in range(n_cols)]
        indices = [(i, j, keys[k]) for i, j, k in all_key_indices if k < n_items]

    return n_rows, n_cols, indices


def remove_duplicate_legend(legend_ax: mpl.axes.Axes) -> None:
    handles, labels = legend_ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    legend_ax.legend(unique.values(), unique.keys())


def add_frame_timestamp(
    ax: mpl.axes.Axes, length: int, width: int, dt: float, frame: int, color: str
) -> None:
    hours, minutes = divmod(round(frame * dt, 2), 1)
    timestamp = f"{int(hours):02d}H:{round(minutes*60):02d}M"

    ax.text(
        0.03 * length,
        0.96 * width,
        timestamp,
        fontfamily="monospace",
        fontsize=20,
        color=color,
        fontweight="bold",
    )


def add_frame_scalebar(
    ax: mpl.axes.Axes, length: int, width: int, ds: float, scale: int, color: str
) -> None:
    scalebar = scale / ds

    ax.add_patch(
        Rectangle(
            (0.95 * length - scalebar, 0.92 * width),
            scalebar,
            0.01 * width,
            snap=True,
            color=color,
        )
    )

    ax.text(
        0.95 * length - scalebar / 2,
        0.975 * width,
        f"{scale} $\\mu$m",
        fontsize=10,
        color=color,
        horizontalalignment="center",
    )
