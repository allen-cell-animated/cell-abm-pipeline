from typing import Optional

import matplotlib.figure as mpl
import pandas as pd
from matplotlib import colormaps
from prefect import task

from cell_abm_pipeline.utilities.plot import make_single_figure


@task
def plot_height_merge(
    keys: list[str],
    data: dict[str, pd.DataFrame],
    reference: pd.DataFrame,
    region: Optional[str] = None,
) -> mpl.Figure:
    fig = make_single_figure()
    cmap = colormaps["tab10"]

    positions = [0] if region is None else [0, 1]
    labels = ["height"] if region is None else ["height", f"height ({region})"]

    ax = fig.add_subplot()

    ax.set_ylabel("Height ($\\mu m$)")
    ax.set_xlim([-0.5, len(positions) - 0.5])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)

    ref_data = reference["height"]

    if region is not None:
        ref_data = reference[["height", f"height.{region}"]]

    ref_violins = ax.violinplot(
        ref_data, positions=positions, showextrema=False, widths=0.8, bw_method=0.25
    )

    for ref_violin in ref_violins["bodies"]:
        ref_violin.set_facecolor("#ccc")
        ref_violin.set_edgecolor("none")
        ref_violin.set_alpha(1)

    ref_means = ref_data.mean(axis=0)
    ax.scatter(positions, ref_means, marker="o", color="#999", s=30, label="reference")

    for index, key in enumerate(keys):
        key_data = data[key]["height"]

        if region is not None:
            key_data = data[key][["height", f"height.{region}"]]

        violins = ax.violinplot(
            key_data, positions=positions, showextrema=False, widths=0.8, bw_method=0.25
        )

        for violin in violins["bodies"]:
            violin.set_facecolor("none")
            violin.set_edgecolor(cmap(index))
            violin.set_linewidth(0.5)
            violin.set_alpha(1)

        means = key_data.mean(axis=0)
        ax.scatter(
            positions,
            means,
            marker="o",
            color="none",
            linewidth=0.5,
            edgecolor=cmap(index),
            s=30,
            zorder=3,
            label=key,
        )

    ax.legend()

    return fig
