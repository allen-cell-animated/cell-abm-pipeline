from typing import Optional

import matplotlib.figure as mpl
import numpy as np
import pandas as pd
from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_volume_distribution(
    keys: list[str],
    data: dict[str, pd.DataFrame],
    bounds: dict[str, tuple[int, int]],
    reference: Optional[pd.DataFrame] = None,
    region: Optional[str] = None,
    thresholds: Optional[list[int]] = None,
) -> mpl.Figure:
    fig, gridspec, indices = make_grid_figure(keys)
    value = f"volume.{region}" if region else "volume"
    bins = np.linspace(*bounds[value], 50)

    for i, j, key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(key)
        ax.set_xlabel("Volume ($\\mu m^3$)")
        ax.set_ylabel("Frequency")

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
            color="k",
            label=f"simulated ({' | '.join(label)})",
        )

        if thresholds is not None:
            for threshold in thresholds:
                subset = data[key][data[key]["time"] >= threshold][value]
                subset_label = [
                    f"{subset.mean():.1f} $\\pm$ {subset.std():.1f} $\\mu m^3$",
                    f"n = {subset.count()}",
                    f"t $\\geq$ {threshold}",
                ]
                ax.hist(
                    subset,
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=0.5,
                    label=f"simulated ({' | '.join(subset_label)})",
                )

        ax.legend()

    return fig
