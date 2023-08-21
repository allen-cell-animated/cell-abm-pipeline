import matplotlib.figure as mpl
import matplotlib.pyplot as plt
import numpy as np
from prefect import task


@task
def make_heatmap_figure(rows: list[str], cols: list[str], values: list[list]) -> mpl.Figure:
    fig = plt.figure(figsize=(4, 4), constrained_layout=True)

    ax = fig.add_subplot()

    ax.imshow(values, cmap="magma_r")

    ax.set_xticks(np.arange(len(cols)), labels=cols)
    ax.set_yticks(np.arange(len(rows)), labels=rows)

    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{values[i][j]:.2f}", ha="center", va="center", color="#999")

    return fig
