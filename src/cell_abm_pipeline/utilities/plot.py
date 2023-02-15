from math import ceil, sqrt

import matplotlib as mpl
import matplotlib.pyplot as plt

PLOT_SIZE = 4

mpl.use("Agg")
mpl.rc("figure", dpi=200)
mpl.rc("font", size=8)
mpl.rc("legend", fontsize=6)
mpl.rc("axes", titlesize=10, titleweight="bold")


def make_single_figure() -> mpl.Figure:
    fig = plt.figure(constrained_layout=True)
    return fig


def make_grid_figure(keys: list[str]) -> mpl.Figure:
    n_rows, n_cols, indices = separate_rows_cols(keys)
    fig = plt.figure(figsize=(n_cols * PLOT_SIZE, n_rows * PLOT_SIZE), constrained_layout=True)
    gridspec = fig.add_gridspec(n_rows, n_cols)
    return fig, gridspec, indices


def separate_rows_cols(keys: list[str]) -> tuple[int, int, list[tuple[int, int, str]]]:
    n_items = len(keys)
    n_cols = ceil(sqrt(n_items))
    n_rows = ceil(n_items / n_cols)

    all_indices = [(i, j, i * n_cols + j) for i in range(n_rows) for j in range(n_cols)]
    indices = [(i, j, keys[k]) for i, j, k in all_indices if k < n_items]

    return n_rows, n_cols, indices


def remove_duplicate_legend(legend_ax: mpl.Axes) -> None:
    handles, labels = legend_ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    legend_ax.legend(unique.values(), unique.keys())
