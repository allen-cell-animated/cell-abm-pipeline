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

    for i, key in enumerate(keys):
        group = clock[clock["KEY"] == key]
        ax.boxplot(group["CLOCK"].values, labels=[key], positions=[i], widths=0.6)
        ax.scatter(
            group["KEY"],
            group["CLOCK"],
            s=10,
            alpha=0.3,
            c="k",
            edgecolors="none",
        )

    return fig
