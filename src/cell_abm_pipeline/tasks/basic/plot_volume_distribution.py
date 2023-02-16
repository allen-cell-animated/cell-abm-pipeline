from typing import Optional

import matplotlib.figure as mpl
import pandas as pd
from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_volume_distribution(
    keys: list[str],
    data: dict[str, pd.DataFrame],
    reference: Optional[pd.DataFrame] = None,
    region: Optional[str] = None,
) -> mpl.Figure:
    fig, gridspec, indices = make_grid_figure(keys)
    value = f"volume.{region}" if region else "volume"
    bins = 50

    for i, j, key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(key)
        ax.set_xlabel("Volume ($\\mu m^3$)")
        ax.set_ylabel("Frequency")

        if reference is not None:
            ref_volumes = reference[value]
            ref_label = (
                f"reference ({ref_volumes.mean():.1f} $\\pm$ {ref_volumes.std():.1f} $\\mu m^3$)"
            )
            ax.hist(
                reference[value],
                bins=bins,
                density=True,
                color="#999999",
                alpha=0.7,
                label=ref_label,
            )

        volumes = data[key][value]
        label = f"simulated ({volumes.mean():.1f} $\\pm$ {volumes.std():.1f} $\\mu m^3$)"
        ax.hist(volumes, bins=bins, density=True, histtype="step", color="k", label=label)

        ax.legend()

    return fig
