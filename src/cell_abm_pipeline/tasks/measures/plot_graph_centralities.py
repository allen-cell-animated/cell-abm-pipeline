import matplotlib.figure as mpl
import pandas as pd
from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure, remove_duplicate_legend


@task
def plot_graph_centralities(keys: list[str], measures: dict[str, pd.DataFrame]) -> mpl.Figure:
    fig, gridspec, indices = make_grid_figure(keys)

    for i, j, key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(key)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Centrality")

        for _, group in measures[key].sort_values(by="TICK").groupby("SEED"):
            time = group["time"] / 24

            degree = group["DEGREE_CENTRALITY_MEAN"]
            closeness = group["CLOSENESS_CENTRALITY_MEAN"]
            betweenness = group["BETWEENNESS_CENTRALITY_MEAN"]

            ax.plot(time, degree, c="#000", alpha=0.5, label="degree")
            ax.plot(time, closeness, c="#900", alpha=0.5, label="closeness")
            ax.plot(time, betweenness, c="#009", alpha=0.5, label="betweenness")

        ax.legend()
        remove_duplicate_legend(ax)

    return fig
