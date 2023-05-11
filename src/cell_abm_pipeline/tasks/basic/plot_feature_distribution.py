from typing import Optional

import matplotlib.figure as mpl
import numpy as np
import pandas as pd
from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_feature_distribution(
    keys: list[str],
    feature: str,
    data: dict[str, pd.DataFrame],
    bin_size: float,
    reference: Optional[pd.DataFrame] = None,
) -> mpl.Figure:
    fig, gridspec, indices = make_grid_figure(keys)

    if "height" in feature:
        unit = "$\\mu m$"
    elif "volume" in feature:
        unit = "$\\mu m^3$"
    else:
        return fig

    bins = get_data_bins(keys, data, bin_size, feature, reference)

    for i, j, key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(key)
        ax.set_xlabel(f"{feature.split('.')[0].title()} ({unit})")

        if reference is not None:
            ref_values = reference[feature]

            ref_label = [
                f"{ref_values.mean():.1f} $\\pm$ {ref_values.std():.1f} {unit}",
                f"n = {ref_values.count()}",
            ]

            ax.hist(
                ref_values,
                bins=bins,
                density=True,
                color="#999999",
                alpha=0.7,
                label=f"reference ({' | '.join(ref_label)})",
            )

        values = data[key][feature]

        label = [
            f"{values.mean():.1f} $\\pm$ {values.std():.1f} {unit}",
            f"n = {values.count()}",
        ]

        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            color="k",
            label=f"simulated ({' | '.join(label)})",
        )

        ax.legend()

    return fig


def get_data_bins(
    keys: list[str],
    data: dict[str, pd.DataFrame],
    bin_size: float,
    feature: str,
    reference: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    if reference is not None:
        ref_values = reference[feature]
        upper_bound = ref_values.max()
    else:
        upper_bound = data[keys[0]][feature].max()

    for key in keys:
        upper_bound = max(upper_bound, data[key][feature].max())

    return np.arange(0, upper_bound + bin_size, bin_size)
