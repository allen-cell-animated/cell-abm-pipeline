import matplotlib.figure as mpl
import pandas as pd
from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_counts_total(keys: list[str], data: dict[str, pd.DataFrame]) -> mpl.Figure:
    fig, gridspec, indices = make_grid_figure(keys)

    for i, j, key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(key)
        ax.set_xlabel("Time (hrs)")
        ax.set_ylabel("Number of cells")

        total_count = data[key].groupby(["SEED", "time"]).size()
        mean = total_count.groupby(["time"]).mean()
        std = total_count.groupby(["time"]).std()
        time = mean.index
        ax.plot(time, mean, c="#000")
        ax.fill_between(time, mean - std, mean + std, facecolor="#bbb")

    return fig
