from typing import Optional

import matplotlib.figure as mpl
import numpy as np
import pandas as pd
from prefect import task
from sklearn.decomposition import PCA

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_variance_explained(
    keys: list[str], models: dict[str, PCA], ref_model: Optional[PCA] = None
) -> mpl.Figure:
    fig, gridspec, indices = make_grid_figure(keys)

    for i, j, key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(f"{key}")
        ax.set_xlabel("Component")
        ax.set_ylabel("Explained variance (%)")

        variance = np.cumsum(models[key].explained_variance_ratio_)
        ax.plot(100 * variance, "-o", color="#555", markersize=3, label="data")

        if ref_model is not None:
            ref_model_variance = np.cumsum(ref_model.explained_variance_ratio_)
            ax.plot(100 * ref_model_variance, "-o", color="#aaa", markersize=3, label="reference")

        ax.set_ylim([0, 100])
        ax.set_xticks(np.arange(0, len(variance), 1))
        ax.set_xticklabels(np.arange(1, len(variance) + 1, 1))
        ax.legend()

    return fig
