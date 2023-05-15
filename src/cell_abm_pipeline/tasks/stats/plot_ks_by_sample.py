import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import colormaps
from prefect import task

from cell_abm_pipeline.utilities.plot import make_single_figure


@task
def plot_ks_by_sample(
    keys: list[str], stats: pd.DataFrame, ordered: bool = True
) -> mpl.figure.Figure:
    fig = make_single_figure()

    if ordered:
        cmap = colormaps["coolwarm"].resampled(len(keys))
    else:
        cmap = colormaps["tab10"]

    ax = fig.add_subplot()
    ax.set_ylabel("Kolmogorov-Smirnov statistic")

    stats_samples = stats[~stats["SAMPLE"].isna()]
    features = stats_samples["FEATURE"].unique()

    bar_width = 1 / (len(keys) + 2)

    for index, key in enumerate(keys):
        key_stats = stats_samples[stats_samples["KEY"] == key].set_index("FEATURE")
        bar_positions = np.arange(len(features)) + index * bar_width
        bar_heights = key_stats.groupby("FEATURE")["KS_STATISTIC"].mean().loc[features]
        std = key_stats.groupby("FEATURE")["KS_STATISTIC"].std(ddof=1).loc[features]
        ax.bar(bar_positions, bar_heights, yerr=std, width=bar_width, label=key, color=cmap(index))

    ax.set_xticks(np.arange(len(features)) + bar_width * (len(keys) / 2 - 0.5), features)
    ax.legend()

    return fig
