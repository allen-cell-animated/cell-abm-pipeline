from typing import Optional

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import colormaps
from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_volume_merge(
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
        ax.set_xlabel("Volume ($\\mu m^3$)")
        ax.set_ylabel("Frequency")

        value = f"volume.{region}" if region_key else "volume"
        bins = np.linspace(*bounds[value], 50)

        if reference is not None:
            ref_volumes = reference[value]
            ref_label = [
                f"{ref_volumes.mean():.1f} $\\pm$ {ref_volumes.std():.1f} $\\mu m^3$",
                f"n = {ref_volumes.count()}",
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
            volumes = data[key][value]

            label = [
                f"{volumes.mean():.1f} $\\pm$ {volumes.std():.1f} $\\mu m^3$",
                f"n = {volumes.count()}",
            ]

            ax.hist(
                volumes,
                bins=bins,
                density=True,
                histtype="step",
                color=cmap(index),
                label=f"simulated {key} ({' | '.join(label)})",
            )

        ax.legend()

    return fig
