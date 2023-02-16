from typing import Optional

import matplotlib.figure as mpl
import pandas as pd
from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_height_distribution(
    keys: list[str],
    data: dict[str, pd.DataFrame],
    reference: Optional[pd.DataFrame] = None,
    region: Optional[str] = None,
) -> mpl.Figure:
    fig, gridspec, indices = make_grid_figure(keys)
    value = f"height.{region}" if region else "height"
    bins = 10

    for i, j, key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(key)
        ax.set_xlabel("Height ($\\mu m$)")
        ax.set_ylabel("Frequency")

        if reference is not None:
            ref_heights = reference[value]
            ref_label = (
                f"reference ({ref_heights.mean():.1f} $\\pm$ {ref_heights.std():.1f} $\\mu m$)"
            )
            ax.hist(
                reference[value],
                bins=bins,
                density=True,
                color="#999999",
                alpha=0.7,
                label=ref_label,
            )

        heights = data[key][value]
        label = f"simulated ({heights.mean():.1f} $\\pm$ {heights.std():.1f} $\\mu m$)"
        ax.hist(heights, bins=bins, density=True, histtype="step", color="k", label=label)

        ax.legend()

    return fig
