import matplotlib.colors as mcolors
import matplotlib.figure as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prefect import task


@task
def make_centroids_figure(
    data: pd.DataFrame,
    frame: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    dt: float,
    window: int,
) -> mpl.Figure:
    fig = plt.figure(figsize=(4, 4), constrained_layout=True)

    ax = fig.add_subplot()

    ax.set_box_aspect(1)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_xlim(xlim)
    ax.set_ylim((ylim[1], ylim[0]))

    ticks = sorted(data["TICK"].unique())
    index = ticks.index(frame)

    lower = ticks[max(0, index - window)]
    upper = ticks[index]
    subset = data[(data["TICK"] <= upper) & (data["TICK"] >= lower)]

    ticks = subset["TICK"].values
    x = subset["CENTER_X"].values
    y = subset["CENTER_Y"].values

    if index == 0:
        sizes = 2 * np.ones((len(ticks)))
        colors = np.ones((len(ticks)))
    else:
        sizes = 1 + (ticks - lower) / (upper - lower)
        colors = (ticks - lower) / (upper - lower)

    colors[subset["PHASE"] == "PROLIFERATIVE_M"] *= -1

    cmap_base = plt.cm.bone_r(np.linspace(0.2, 1, 128))
    cmap_accent = plt.cm.Reds_r(np.linspace(0.2, 1, 128))
    cmap_combined = np.vstack((cmap_accent, cmap_base))
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", cmap_combined)

    sizes = sizes**2
    ax.scatter(x, y, s=sizes, c=colors, cmap=cmap, vmin=-1, vmax=1)

    hours, minutes = divmod(round(upper * dt, 2), 1)
    timestamp = f"{int(hours):02d}H:{round(minutes*60):02d}M"

    ax.text(
        0.02,
        0.02,
        timestamp,
        fontfamily="monospace",
        fontsize=20,
        fontweight="bold",
        transform=ax.transAxes,
    )

    return fig
