import matplotlib.figure as mpl
import numpy as np
import pandas as pd
from prefect import task
from sklearn.decomposition import PCA

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_feature_compare(
    keys: list[str], feature: str, data: dict[str, pd.DataFrame], ref_data: pd.DataFrame
) -> mpl.Figure:
    fig, gridspec, indices = make_grid_figure(keys)

    ref_feature_data = ref_data[feature]

    if "height" in feature:
        bins = np.linspace(ref_feature_data.min(), ref_feature_data.max(), 10)
    else:
        bins = np.linspace(ref_feature_data.min(), ref_feature_data.max(), 50)

    for i, j, key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(key)
        ax.set_xlabel(feature)
        key_data = data[key][feature]

        ax.hist(
            ref_feature_data,
            bins=bins,
            density=True,
            alpha=0.3,
            color="black",
            label=f"reference (n = {ref_feature_data.shape[0]})",
        )
        ax.hist(
            key_data,
            bins=bins,
            color="black",
            density=True,
            histtype="step",
            label=f"simulated (n = {key_data.shape[0]})",
        )

        ax.legend()

    return fig
