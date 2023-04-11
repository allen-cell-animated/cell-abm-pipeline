import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import colormaps
from prefect import task

from cell_abm_pipeline.utilities.plot import make_single_figure


@task
def plot_ks_by_sample(keys: list[str], stats: pd.DataFrame) -> mpl.figure.Figure:
    fig = make_single_figure()
    cmap = colormaps["tab20"]

    ax = fig.add_subplot()
    ax.set_xlabel("Key")
    ax.set_ylabel("Kolmogorov-Smirnov statistic")

    stats_samples = stats[~stats["SAMPLE"].isna()]

    for index, (feature, group) in enumerate(stats_samples.groupby("FEATURE")):
        summary = group.groupby("KEY")["KS_STATISTIC"]
        means = summary.mean()
        stds = summary.std()

        min_deltas = [means[key] - summary.min()[key] for key in keys]
        max_deltas = [summary.max()[key] - means[key] for key in keys]

        ax.errorbar(
            keys,
            [means[key] for key in keys],
            yerr=np.vstack((min_deltas, max_deltas)),
            marker="o",
            markersize=4,
            linestyle="dashed",
            linewidth=0.5,
            color=cmap(index),
            label=feature,
        )

        ax.errorbar(
            keys,
            [means[key] for key in keys],
            yerr=[stds[key] for key in keys],
            marker="",
            linestyle="",
            linewidth=2,
            color=cmap(index),
        )

    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    return fig
