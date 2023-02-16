import matplotlib.figure as mpl
import pandas as pd
from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_degree_trajectory(
    keys: list[str], measures: dict[str, pd.DataFrame], summary: str
) -> mpl.Figure:
    fig, gridspec, indices = make_grid_figure(keys)

    for i, j, key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(key)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel(summary.title().replace("_", " "))

        for _, group in measures[key].sort_values(by="TICK").groupby("SEED"):
            time = group["time"] / 24
            summary_value = group[summary]
            ax.plot(time, summary_value, c="#000", alpha=0.5)

    return fig
