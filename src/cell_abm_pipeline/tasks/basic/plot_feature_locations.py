from typing import Optional

import matplotlib as mpl
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_feature_locations(
    keys: list[tuple[str, int]],
    feature: str,
    data: dict[tuple[str, int], pd.DataFrame],
    tick: int = 0,
    reference: Optional[pd.DataFrame] = None,
) -> mpl.figure.Figure:
    fig, gridspec, indices = make_grid_figure(keys)
    min_x, max_x, min_y, max_y, padding = get_location_bounds(data)

    if reference is not None:
        reference_value = reference[feature].mean()

    for i, j, (key, seed) in indices:
        ax = fig.add_subplot(gridspec[i, j])
        format_location_axis(ax, f"{key} [{seed}]", min_x, max_x, min_y, max_y, padding)

        key_seed_data = data[(key, seed)]
        tick_data = key_seed_data[key_seed_data["TICK"] == tick]

        x = tick_data["CENTER_X"]
        y = tick_data["CENTER_Y"]

        if reference is not None:
            values = (tick_data[feature] - reference_value) / reference_value
            sax = ax.scatter(x, y, c=values, s=20, cmap="coolwarm", vmin=-1, vmax=1)
        else:
            values = tick_data[feature]
            sax = ax.scatter(x, y, c=values, s=20, cmap="magma_r")

        cbax = inset_axes(ax, width="3%", height="96%", loc="upper right")
        colorbar = fig.colorbar(sax, cax=cbax)
        colorbar.ax.yaxis.set_ticks_position("left")

    return fig


def get_location_bounds(
    data: dict[tuple[str, int], pd.DataFrame]
) -> tuple[int, int, int, int, float]:
    all_data = pd.concat(data.values())

    max_x = all_data["CENTER_X"].max()
    min_x = all_data["CENTER_X"].min()
    max_y = all_data["CENTER_Y"].max()
    min_y = all_data["CENTER_Y"].min()
    padding = 0.5 * max((max_x - min_x), (max_y - min_y))

    return min_x, max_x, min_y, max_y, padding


def format_location_axis(
    ax: mpl.axes.Axes, title: str, min_x: int, max_x: int, min_y: int, max_y: int, padding: float
) -> None:
    ax.set_title(title)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_xlim((min_x - padding, max_x + padding))
    ax.set_ylim((min_y - padding, max_y + padding))
    ax.invert_yaxis()
