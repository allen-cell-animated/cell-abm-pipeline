from typing import Optional

import matplotlib.figure as mpl
import pandas as pd
from matplotlib.lines import Line2D
from prefect import task

from cell_abm_pipeline.utilities.plot import (
    add_frame_scalebar,
    add_frame_timestamp,
    make_grid_figure,
)


@task
def make_graph_frame(
    keys: list[tuple[str, int]],
    neighbors: dict[tuple[str, int], pd.DataFrame],
    tick: int,
    scale: int,
    ds: float,
    dt: float,
    box: tuple[int, int, int],
    phase_colors: Optional[dict[str, str]],
) -> mpl.Figure:
    fig, gridspec, indices = make_grid_figure(keys)
    length, width, _ = box

    handles = None
    if phase_colors is not None:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=phase,
                markerfacecolor=color,
                markersize=5,
            )
            for phase, color in phase_colors.items()
        ]

    for i, j, (key, seed) in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(f"{key} [{seed}]")
        ax.invert_yaxis()
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_xlim([0, length - 1])
        ax.set_ylim([width - 1, 0])

        for spine in ax.spines.values():
            spine.set_edgecolor("#dddddd")

        key_seed_neighbors = neighbors[(key, seed)]
        frame = key_seed_neighbors[key_seed_neighbors["TICK"] == tick]
        x_centroids = dict(zip(frame["ID"], frame["CX"]))
        y_centroids = dict(zip(frame["ID"], frame["CY"]))

        edges = {
            tuple(sorted([node_id, neighbor_id]))
            for node_id, neighbor_ids in zip(frame["ID"], frame["NEIGHBORS"])
            for neighbor_id in neighbor_ids
        }

        for from_id, to_id in edges:
            ax.plot(
                [x_centroids[from_id], x_centroids[to_id]],
                [y_centroids[from_id], y_centroids[to_id]],
                color="k",
                lw=0.5,
                zorder=1,
            )

        if phase_colors is None:
            colors = list(frame["DEPTH"])
        else:
            colors = [phase_colors[phase] for phase in frame["PHASE"]]
            ax.legend(handles=handles, loc="upper right")

        ecolors = ["#000" if depth == 1 else "#fff" for depth in frame["DEPTH"]]
        widths = [1 if depth == 1 else 0.2 for depth in frame["DEPTH"]]
        ax.scatter(frame["CX"], frame["CY"], c=colors, s=20, zorder=2, edgecolor=ecolors, lw=widths)

        add_frame_timestamp(ax, length, width, dt, tick, "#000000")
        add_frame_scalebar(ax, length, width, ds, scale, "#000000")

    return fig
