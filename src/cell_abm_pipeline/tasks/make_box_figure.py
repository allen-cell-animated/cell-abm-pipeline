import matplotlib.figure as mpl
import matplotlib.pyplot as plt
import pandas as pd
from prefect import task


@task
def make_box_figure(
    keys: list[str], data: pd.DataFrame, xlabel: str = "", ylabel: str = ""
) -> mpl.Figure:
    fig = plt.figure(figsize=(4, 4), constrained_layout=True)

    ax = fig.add_subplot()
    ax.set_box_aspect(1)

    for index, key in enumerate(keys):
        group = data[data["key"] == key]

        ax.boxplot(group["value"].values, labels=[key], positions=[index], widths=0.6)
        ax.scatter(
            group["key"],
            group["value"],
            s=10,
            alpha=0.3,
            c="k",
            edgecolors="none",
        )

    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")

    return fig
