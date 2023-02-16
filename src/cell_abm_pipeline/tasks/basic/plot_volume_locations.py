from typing import Optional

import matplotlib.figure as mpl
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_volume_locations(
    keys: list[tuple[str, int]],
    data: dict[tuple[str, int], pd.DataFrame],
    tick: int = 0,
    reference: Optional[pd.DataFrame] = None,
    region: Optional[str] = None,
) -> mpl.Figure:
    fig, gridspec, indices = make_grid_figure(keys)
    value = f"volume.{region}" if region else "volume"

    all_data = pd.concat(data.values())
    max_x = all_data["CENTER_X"].max()
    min_x = all_data["CENTER_X"].min()
    max_y = all_data["CENTER_Y"].max()
    min_y = all_data["CENTER_Y"].min()
    padding = 0.5 * max((max_x - min_x), (max_y - min_y))

    if reference is not None:
        reference_volume = reference[value].mean()
    else:
        reference_volume = all_data[all_data["TICK"] == tick][value].mean()

    for i, j, (key, seed) in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(f"{key} [{seed}]")
        ax.invert_yaxis()
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_xlim((min_x - padding, max_x + padding))
        ax.set_ylim((min_y - padding, max_y + padding))

        key_seed_data = data[(key, seed)]
        tick_data = key_seed_data[key_seed_data["TICK"] == tick]

        x = tick_data["CENTER_X"]
        y = tick_data["CENTER_Y"]
        volumes = (tick_data[value] - reference_volume) / reference_volume

        sax = ax.scatter(x, y, c=volumes, s=20, cmap="coolwarm", vmin=-1, vmax=1)
        cbax = inset_axes(ax, width="3%", height="96%", loc="upper right")
        colorbar = fig.colorbar(sax, cax=cbax)
        colorbar.ax.yaxis.set_ticks_position("left")

    return fig
