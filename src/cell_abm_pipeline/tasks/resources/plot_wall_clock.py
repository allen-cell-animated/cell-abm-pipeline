import matplotlib.figure as mpl
import pandas as pd
from prefect import task

from cell_abm_pipeline.utilities.plot import make_single_figure


@task
def plot_wall_clock(keys: list[str], clock: pd.DataFrame) -> mpl.Figure:
    fig = make_single_figure()

    ax = fig.add_subplot()
    ax.set_ylabel("Wall clock time (minutes)")
    ax.set_xlim([-0.5, len(keys) - 0.5])

    values = [clock[clock["KEY"] == key]["CLOCK"].values for key in keys]
    ax.boxplot(values, labels=keys, positions=range(0, len(keys)))
    ax.scatter(clock["KEY"], clock["CLOCK"], s=10, alpha=0.3, c="k", edgecolors="none")

    return fig
