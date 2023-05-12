import matplotlib.figure as mpl
import pandas as pd
from matplotlib.lines import Line2D
from prefect import task

from cell_abm_pipeline.tasks.basic.plot_feature_locations import (
    format_location_axis,
    get_location_bounds,
)
from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_phase_locations(
    keys: list[tuple[str, int]],
    data: dict[tuple[str, int], pd.DataFrame],
    tick: int,
    phase_colors: dict[str, str],
) -> mpl.Figure:
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

    fig, gridspec, indices = make_grid_figure(keys)
    min_x, max_x, min_y, max_y, padding = get_location_bounds(data)

    for i, j, (key, seed) in indices:
        ax = fig.add_subplot(gridspec[i, j])
        format_location_axis(ax, f"{key} [{seed}]", min_x, max_x, min_y, max_y, padding)

        key_seed_data = data[(key, seed)]
        tick_data = key_seed_data[key_seed_data["TICK"] == tick]

        x = tick_data["CENTER_X"]
        y = tick_data["CENTER_Y"]
        phases = [phase_colors[phase] for phase in tick_data["PHASE"]]

        ax.scatter(x, y, c=phases, s=20)
        ax.legend(handles=handles, loc="upper right")

    return fig
