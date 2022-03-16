from math import sqrt, ceil, floor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
from matplotlib.lines import Line2D

PLOT_SIZE = 2

rc("font", size=8)
rc("axes", titlesize=10, titleweight="bold")


def make_plot(keys, data, func, size=PLOT_SIZE, xlabel="", ylabel="", legend=False):
    n_rows, n_cols, indices = separate_keys(keys)
    offset = size if legend else 0
    fig, axs = make_subplots(n_rows, n_cols, size=size, offset=offset)

    for i, j, k in indices:
        key = keys[k]
        ax = select_axes(axs, i, j, n_rows, n_cols)
        func(ax, data, key)
        ax.set_title(key)

    col_ax = select_axes(axs, -1, floor(n_cols / 2), n_rows, n_cols)
    col_ax.set_xlabel(xlabel)

    row_ax = select_axes(axs, floor(n_rows / 2), 0, n_rows, n_cols)
    row_ax.set_ylabel(ylabel)

    if legend:
        legend_ax = select_axes(axs, 0, -1, n_rows, n_cols)

        if isinstance(legend, dict):
            legend_ax.legend(**legend, bbox_to_anchor=(1.2, 1), loc="upper left")
        else:
            legend_ax.legend(bbox_to_anchor=(1.2, 1), loc="upper left")

    fig.tight_layout()


def make_subplots(n_rows, n_cols, size, offset=0, sharex="all", sharey="all"):
    figsize = (n_cols * size + offset, n_rows * size)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=sharex, sharey=sharey)
    return fig, axs


def separate_keys(keys):
    n_keys = len(keys)
    n_cols = ceil(sqrt(len(keys)))
    n_rows = ceil(len(keys) / n_cols)

    indices = [(i, j, i * n_cols + j) for i in range(n_rows) for j in range(n_cols)]
    indices = [(i, j, k) for i, j, k in indices if k < n_keys]

    return n_rows, n_cols, indices


def select_axes(axs, i, j, n_rows, n_cols):
    if n_rows == 1 and n_cols == 1:
        return axs
    elif n_rows == 1:
        return axs[j]
    else:
        return axs[i, j]


def make_legend(label, bounds, intervals=5, colormap="magma_r"):
    cmap = cm.get_cmap(colormap)
    elements = []

    for i in np.linspace(0, 1, intervals):
        value = int(bounds[0] + i * bounds[1])
        elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"{label} = {value}",
                markerfacecolor=cmap(i),
                markersize=5,
            )
        )

    return elements
