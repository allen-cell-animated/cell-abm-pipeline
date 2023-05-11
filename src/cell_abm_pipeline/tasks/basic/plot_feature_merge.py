from typing import Optional

import matplotlib.figure as mpl
import numpy as np
import pandas as pd
from matplotlib import colormaps
from prefect import task

from cell_abm_pipeline.tasks.basic.plot_feature_distribution import get_data_bins
from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_feature_merge(
    keys: list[str],
    feature: str,
    data: dict[str, pd.DataFrame],
    bin_sizes: dict[str, float],
    reference: Optional[pd.DataFrame] = None,
    region: Optional[str] = None,
    ordered: bool = True,
) -> mpl.Figure:
    region_keys: list[Optional[str]] = [None] if region is None else [None, region]
    fig, gridspec, indices = make_grid_figure(region_keys)

    if ordered:
        cmap = colormaps["coolwarm"].resampled(len(keys))
    else:
        cmap = colormaps["tab10"]

    if "height" in feature:
        unit = "$\\mu m$"
    elif "volume" in feature:
        unit = "$\\mu m^3$"
    else:
        return fig

    for i, j, region_key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(region_key)
        ax.set_xlabel(f"{feature.title()} ({unit})")

        feature_name = f"{feature}.{region}" if region_key else feature
        bin_size = bin_sizes[feature_name]
        bins = get_data_bins(keys, data, bin_size, feature_name, reference)

        if reference is not None:
            ref_values = reference[feature_name]

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

        for index, key in enumerate(keys):
            values = data[key][feature_name]

            label = [
                f"{values.mean():.1f} $\\pm$ {values.std():.1f} {unit}",
                f"n = {values.count()}",
            ]

            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                color=cmap(index),
                label=f"simulated {key} ({' | '.join(label)})",
            )

        ax.legend()

    return fig
