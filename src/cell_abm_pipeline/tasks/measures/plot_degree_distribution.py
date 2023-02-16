import matplotlib.figure as mpl
import numpy as np
import pandas as pd
from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_degree_distribution(keys: list[str], measures: dict[str, pd.DataFrame]) -> mpl.Figure:
    fig, gridspec, indices = make_grid_figure(keys)

    for i, j, key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(key)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Frequency")

        final_tick = measures[key]["TICK"].max()
        degree_data = measures[key][measures[key]["TICK"] == final_tick]
        degrees = [deg for degree in degree_data["DEGREES"] for deg in degree]

        ax.bar(*np.unique(degrees, return_counts=True))

    return fig
