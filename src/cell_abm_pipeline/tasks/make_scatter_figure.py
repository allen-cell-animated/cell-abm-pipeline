from typing import Optional

import matplotlib.figure as mpl
import matplotlib.pyplot as plt
import pandas as pd
from prefect import task


@task
def make_scatter_figure(data: pd.DataFrame, colormap: Optional[dict] = None) -> mpl.Figure:
    fig = plt.figure(figsize=(4, 4), constrained_layout=True)

    ax = fig.add_subplot()
    ax.set_box_aspect(1)

    if colormap is not None:
        ax.scatter(data["x"], data["y"], c=data["v"].map(colormap), s=10)
    else:
        ax.scatter(data["x"], data["y"], c=data["v"], cmap="magma_r", s=10)

    return fig
