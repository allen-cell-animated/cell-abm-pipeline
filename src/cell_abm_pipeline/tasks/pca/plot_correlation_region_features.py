import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from prefect import task

from cell_abm_pipeline.tasks.pca.plot_correlation_all_features import (
    plot_linear_fit,
    plot_pearson_statistic,
    plot_rolling_average,
)
from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_correlation_region_features(data: pd.DataFrame, regions: list[str]) -> mpl.figure.Figure:

    features = [
        f"{feature}.{region}" if region != "DEFAULT" else feature
        for feature in [
            "volume",
            "height",
            "area",
            "axis_major_length",
            "axis_minor_length",
            "eccentricity",
            "orientation",
            "perimeter",
        ]
        for region in regions
    ]

    feature_combos = [
        (feature_a, feature_b)
        for index_a, feature_a in enumerate(features)
        for index_b, feature_b in enumerate(features)
        if index_a < index_b
    ]

    fig, gridspec, indices = make_grid_figure(feature_combos, size=2)

    row_ax = {}
    col_ax = {}

    for i, j, (key_a, key_b) in indices:
        if i == 0 and j == 0:
            ax = fig.add_subplot(gridspec[i, j])
            row_ax[i] = ax
            col_ax[j] = ax
        elif j == 0:
            ax = fig.add_subplot(gridspec[i, j], sharex=col_ax[j])
            row_ax[i] = ax
        elif i == j:
            ax = fig.add_subplot(gridspec[i, j], sharey=row_ax[i])
            col_ax[i] = ax
        else:
            ax = fig.add_subplot(gridspec[i, j], sharex=col_ax[j], sharey=row_ax[i])

        if j == 0:
            ax.set_ylabel(key_b)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)

        if i == len(features) - 2:
            ax.set_xlabel(key_a)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)

        data_a = data[key_a]
        data_b = data[key_b]

        ax.scatter(data_a, data_b, color="k", alpha=0.3, s=2)

        plot_linear_fit(data_a, data_b, ax)
        plot_rolling_average(data_a, data_b, ax)
        plot_pearson_statistic(data_a, data_b, ax)

    return fig
