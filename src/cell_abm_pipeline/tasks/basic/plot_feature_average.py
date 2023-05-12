from typing import Optional

import matplotlib.figure as mpl
import pandas as pd
from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_feature_average(
    keys: list[str],
    feature: str,
    data: dict[str, pd.DataFrame],
    reference: Optional[pd.DataFrame] = None,
) -> mpl.Figure:
    fig, gridspec, indices = make_grid_figure(keys)

    if "height" in feature:
        unit = "$\\mu m$"
    elif "volume" in feature:
        unit = "$\\mu m^3$"
    else:
        return fig

    for i, j, key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(key)
        ax.set_xlabel("Time (hrs)")
        ax.set_ylabel(f"Average {feature.split('.')[0]} ({unit})")

        values = data[key].groupby(["SEED", "time"])[feature].mean()
        mean = values.groupby(["time"]).mean()
        std = values.groupby(["time"]).std()
        time = mean.index

        if reference is not None:
            ref_values_mean = reference[feature].mean()
            ref_values_std = reference[feature].std()
            ref_label = f"reference ({ref_values_mean:.1f} $\\pm$ {ref_values_std:.1f} $\\mu m^3$)"
            ax.plot(
                time,
                [reference[feature].mean()] * len(time),
                c="#555",
                lw=0.5,
                label=ref_label,
            )

        label = f"simulated ({mean.mean():.1f} $\\pm$ {std.mean():.1f} $\\mu m^3$)"
        ax.plot(time, mean, c="#000", label=label)
        ax.fill_between(time, mean - std, mean + std, facecolor="#bbb")

        ax.legend()

    return fig
