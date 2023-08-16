from math import sqrt

import matplotlib.figure as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.patches import RegularPolygon
from prefect import task


@task
def make_density_figure(data: pd.DataFrame, scale: float) -> mpl.Figure:
    fig = plt.figure(figsize=(4, 4), constrained_layout=True)

    ax = fig.add_subplot()
    ax.set_box_aspect(1)

    cmap = colormaps["magma_r"]
    data["vn"] = (data["v"] - data["v"].min()) / (data["v"].max() - data["v"].min())

    for _, row in data.iterrows():
        hexagon = RegularPolygon(
            (row["x"], row["y"]),
            numVertices=6,
            radius=(scale / sqrt(3)),
            orientation=np.radians(30),
            facecolor=cmap(row["vn"]),
        )
        ax.add_patch(hexagon)

    ax.set_xbound([data["x"].min() - scale, data["x"].max() + scale])
    ax.set_ybound([data["y"].min() - scale, data["y"].max() + scale])

    return fig
