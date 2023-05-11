from typing import Optional

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import colormaps
from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_height_merge(
    keys: list[str],
    data: dict[str, pd.DataFrame],
    bounds: dict[str, tuple[int, int]],
    reference: pd.DataFrame,
    region: Optional[str] = None,
) -> mpl.figure.Figure:
    region_keys: list[Optional[str]] = [None] if region is None else [None, region]
    fig, gridspec, indices = make_grid_figure(region_keys)
    cmap = colormaps["tab10"]

    for i, j, region_key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(region_key)
        ax.set_xlabel("Height ($\\mu m$)")
        ax.set_ylabel("Frequency")

        value = f"height.{region}" if region_key else "height"
        bins = np.linspace(*bounds[value], 20)

        if reference is not None:
            ref_heights = reference[value]
            ref_label = [
                f"{ref_heights.mean():.1f} $\\pm$ {ref_heights.std():.1f} $\\mu m$",
                f"n = {ref_heights.count()}",
            ]
            ax.hist(
                reference[value],
                bins=bins,
                density=True,
                color="#999999",
                alpha=0.7,
                label=f"reference ({' | '.join(ref_label)})",
            )

        for index, key in enumerate(keys):
            heights = data[key][value]

            label = [
                f"{heights.mean():.1f} $\\pm$ {heights.std():.1f} $\\mu m$",
                f"n = {heights.count()}",
            ]

            ax.hist(
                heights,
                bins=bins,
                density=True,
                histtype="step",
                color=cmap(index),
                label=f"simulated {key} ({' | '.join(label)})",
            )

        ax.legend()

    return fig
