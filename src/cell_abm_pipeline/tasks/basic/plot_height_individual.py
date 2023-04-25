from typing import Optional

import matplotlib.figure as mpl
import pandas as pd
from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_height_individual(
    keys: list[str],
    data: dict[str, pd.DataFrame],
    region: Optional[str] = None,
    ids: Optional[list[int]] = None,
) -> mpl.Figure:
    fig, gridspec, indices = make_grid_figure(keys)
    value = f"height.{region}" if region else "height"

    for i, j, key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(key)
        ax.set_xlabel("Time (hrs)")
        ax.set_ylabel("Height ($\\mu m$)")

        key_data = data[key]

        if ids is not None:
            key_data = key_data[key_data["ID"].isin(ids)]

        for _, group in key_data.groupby(["SEED", "ID"]):
            group.sort_values("time", inplace=True)

            height = group[value].values
            time = group["time"]

            ax.plot(time, height, lw=0.5, alpha=0.5)

    return fig
