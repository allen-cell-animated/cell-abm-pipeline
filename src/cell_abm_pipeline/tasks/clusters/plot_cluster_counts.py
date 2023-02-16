import matplotlib.figure as mpl
import pandas as pd
from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure, remove_duplicate_legend


@task
def plot_cluster_counts(keys: list[str], clusters: dict[str, pd.DataFrame]) -> mpl.Figure:
    fig, gridspec, indices = make_grid_figure(keys)

    for i, j, key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(key)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Number")

        for _, group in clusters[key].sort_values(by="TICK").groupby("SEED"):
            time = group["time"] / 24

            num_clusters = group["NUM_CLUSTERS"]
            num_singles = group["NUM_SINGLES"]

            ax.plot(time, num_clusters, c="#f00", alpha=0.5, label="clusters")
            ax.plot(time, num_singles, c="#00f", alpha=0.5, label="singles")

        ax.legend()
        remove_duplicate_legend(ax)

    return fig
