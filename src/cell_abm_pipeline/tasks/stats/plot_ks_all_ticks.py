from typing import Optional

import matplotlib as mpl
import pandas as pd
from matplotlib import colormaps
from prefect import task

from cell_abm_pipeline.utilities.plot import make_single_figure


@task
def plot_ks_all_ticks(
    keys: list[str], stats: pd.DataFrame, ref_stats: Optional[pd.DataFrame] = None
) -> mpl.figure.Figure:
    fig = make_single_figure()
    cmap = colormaps["tab20"]

    ax = fig.add_subplot()
    ax.set_xlabel("Key")
    ax.set_ylabel("Kolmogorov-Smirnov statistic")

    stats_all_ticks = stats[stats["TICK"].isna() & stats["SAMPLE"].isna()]

    for index, (feature, group) in enumerate(stats_all_ticks.groupby("FEATURE")):
        ax.plot(
            group["KEY"],
            group["KS_STATISTIC"],
            marker="o",
            markersize=5,
            linestyle="dashed",
            linewidth=0.5,
            color=cmap(index),
            label=feature,
        )

    if ref_stats is not None:
        ref_stats_all_ticks = ref_stats[ref_stats["TICK"].isna() & ref_stats["SAMPLE"].isna()]
        for index, (feature, group) in enumerate(ref_stats_all_ticks.groupby("FEATURE")):
            value = group["KS_STATISTIC"]
            ax.plot([-0.2], [value], marker=">", color=cmap(index), markersize=5)
            ax.plot([len(keys) - 0.8], [value], marker="<", color=cmap(index), markersize=5)

    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    return fig
