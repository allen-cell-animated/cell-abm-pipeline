from typing import Optional

import matplotlib.figure as mpl
import numpy as np
import pandas as pd
from prefect import task
from sklearn.decomposition import PCA

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_transform_compare(
    keys: list[str],
    component: int,
    model: PCA,
    data: dict[str, pd.DataFrame],
    ref_data: pd.DataFrame,
    thresholds: Optional[list[int]] = None,
) -> mpl.Figure:
    fig, gridspec, indices = make_grid_figure(keys)

    columns = ref_data.filter(like="shcoeffs").columns
    ref_transform = model.transform(ref_data[columns].values)

    for i, j, key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(f"{key}")
        ax.set_xlabel(f"PC {component + 1}")

        bins = np.linspace(ref_transform[:, component].min(), ref_transform[:, component].max(), 50)
        key_transform = model.transform(data[key][columns].values)

        ax.hist(
            ref_transform[:, component],
            bins=bins,
            density=True,
            alpha=0.3,
            color="black",
            label=f"reference (n = {ref_transform.shape[0]})",
        )
        ax.hist(
            key_transform[:, component],
            bins=bins,
            color="black",
            density=True,
            histtype="step",
            label=f"simulated (n = {key_transform.shape[0]})",
        )

        if thresholds is not None:
            for threshold in thresholds:
                subset = data[key][data[key]["time"] >= threshold]
                subset_transform = model.transform(subset[columns].values)
                ax.hist(
                    subset_transform[:, component],
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=0.5,
                    label=f"simulated (n = {subset_transform.shape[0]} | t $\\geq$ {threshold} hrs)",
                )

        ax.legend()

    return fig
