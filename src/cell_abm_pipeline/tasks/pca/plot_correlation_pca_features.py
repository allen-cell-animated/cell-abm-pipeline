import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from prefect import task
from sklearn.decomposition import PCA

from cell_abm_pipeline.tasks.pca.plot_correlation_all_features import (
    plot_linear_fit,
    plot_pearson_statistic,
    plot_rolling_average,
)
from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_correlation_pca_features(model: PCA, data: pd.DataFrame) -> mpl.figure.Figure:
    columns = data.filter(like="shcoeffs").columns
    transform = model.transform(data[columns].values)
    components = transform.shape[1]

    feature_combos = [
        (feature_a, feature_b)
        for index_a, feature_a in enumerate(range(components))
        for index_b, feature_b in enumerate(range(components))
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
            ax.set_ylabel(f"PC {key_b + 1}")
        else:
            plt.setp(ax.get_yticklabels(), visible=False)

        if i == components - 2:
            ax.set_xlabel(f"PC {key_a + 1}")
        else:
            plt.setp(ax.get_xticklabels(), visible=False)

        data_a = transform[:, key_a]
        data_b = transform[:, key_b]

        ax.scatter(data_a, data_b, color="k", alpha=0.3, s=2)

        plot_linear_fit(data_a, data_b, ax)
        plot_rolling_average(data_a, data_b, ax)
        plot_pearson_statistic(data_a, data_b, ax)

    return fig
