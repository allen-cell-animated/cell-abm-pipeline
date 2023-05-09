import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prefect import task
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_correlation_all_features(
    model: PCA, data: pd.DataFrame, regions: list[str]
) -> mpl.figure.Figure:
    columns = data.filter(like="shcoeffs").columns
    transform = model.transform(data[columns].values)
    components = transform.shape[1]

    pca_features = list(range(components))
    region_features = [
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
        (feature_a, feature_b) for feature_a in pca_features for feature_b in region_features
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
        elif i == 0:
            ax = fig.add_subplot(gridspec[i, j], sharey=row_ax[i])
            col_ax[j] = ax
        else:
            ax = fig.add_subplot(gridspec[i, j], sharex=col_ax[j], sharey=row_ax[i])

        if j == 0:
            ax.set_ylabel(key_b)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)

        if i == len(region_features) - 1:
            ax.set_xlabel(f"PC {key_a + 1}")
        else:
            plt.setp(ax.get_xticklabels(), visible=False)

        data_a = transform[:, key_a]
        data_b = data[key_b]

        ax.scatter(data_a, data_b, color="k", alpha=0.3, s=2)

        plot_linear_fit(data_a, data_b, ax)
        plot_rolling_average(data_a, data_b, ax)
        plot_pearson_statistic(data_a, data_b, ax)

    return fig


def plot_linear_fit(data_a: np.ndarray, data_b: np.ndarray, ax: mpl.axes.Axes) -> None:
    slope, intercept = np.polyfit(data_a, data_b, 1)
    ax.plot(data_a, intercept + slope * data_a, linewidth=0.5, linestyle="dotted")
    ax.scatter(data_a, data_b, color="k", alpha=0.3, s=2)


def plot_rolling_average(data_a: np.ndarray, data_b: np.ndarray, ax: mpl.axes.Axes) -> None:
    paired = list(zip(data_a, data_b))
    paired.sort(key=lambda x: x[0])
    paired_array = np.array(paired)
    convolve_vector = np.ones(1000) / 1000
    means_a = np.convolve(paired_array[:, 0], convolve_vector, mode="valid")
    means_b = np.convolve(paired_array[:, 1], convolve_vector, mode="valid")
    ax.plot(means_a, means_b, c="red", lw=2)


def plot_pearson_statistic(data_a: np.ndarray, data_b: np.ndarray, ax: mpl.axes.Axes) -> None:
    pearson = pearsonr(data_a, data_b).statistic

    if abs(pearson) > 0.5:
        ax.text(0.05, 0.9, f"p = {pearson:.2f}", transform=ax.transAxes, fontweight="bold")
    elif abs(pearson) > 0.1:
        ax.text(0.05, 0.9, f"p = {pearson:.2f}", transform=ax.transAxes)
    else:
        ax.text(0.05, 0.9, f"p = {pearson:.2f}", transform=ax.transAxes, color="#999")
