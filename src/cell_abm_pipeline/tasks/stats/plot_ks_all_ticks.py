import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import colormaps
from prefect import task

from cell_abm_pipeline.utilities.plot import make_single_figure


@task
def plot_ks_all_ticks(
    keys: list[str], stats: pd.DataFrame, ordered: bool = True
) -> mpl.figure.Figure:
    fig = make_single_figure()

    if ordered:
        cmap = colormaps["coolwarm"].resampled(len(keys))
    else:
        cmap = colormaps["tab10"]

    ax = fig.add_subplot()
    ax.set_ylabel("Kolmogorov-Smirnov statistic")

    stats_all_ticks = stats[stats["TICK"].isna() & stats["SAMPLE"].isna()]
    features = stats_all_ticks["FEATURE"].unique()

    bar_width = 1 / (len(keys) + 2)

    for index, key in enumerate(keys):
        key_stats = stats_all_ticks[stats_all_ticks["KEY"] == key].set_index("FEATURE")
        bar_positions = np.arange(len(features)) + index * bar_width
        bar_heights = key_stats.loc[features]["KS_STATISTIC"]
        ax.bar(bar_positions, bar_heights, width=bar_width, label=key, color=cmap(index))

    ax.set_xticks(np.arange(len(features)) + bar_width * (len(keys) / 2 - 0.5), features)
    ax.legend()

    return fig
