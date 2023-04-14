import matplotlib.figure as mpl
import pandas as pd
from matplotlib import colormaps
from prefect import task
from sklearn.decomposition import PCA

from cell_abm_pipeline.utilities.plot import make_single_figure


@task
def plot_feature_merge(
    keys: list[str],
    feature: str,
    data: dict[str, pd.DataFrame],
    ref_data: pd.DataFrame,
    regions: list[str],
    ordered: bool = True,
) -> mpl.Figure:
    fig = make_single_figure()

    if ordered:
        cmap = colormaps["coolwarm"].resampled(len(keys))
    else:
        cmap = colormaps["tab20"]

    feature_names = [
        f"{feature}.{region}" if region != "DEFAULT" else feature for region in regions
    ]
    positions = range(len(regions))

    ax = fig.add_subplot()

    ax.set_xlim([-0.5, len(regions) - 0.5])
    ax.set_xticks(positions)
    ax.set_xticklabels(regions)

    # for position, region in enumerate(regions):
    ref_violins = ax.violinplot(
        ref_data[feature_names], positions=positions, showextrema=False, widths=0.8
    )

    for ref_violin in ref_violins["bodies"]:
        ref_violin.set_facecolor("#ddd")
        ref_violin.set_edgecolor("none")
        ref_violin.set_alpha(1)

    for index, key in enumerate(keys):
        violins = ax.violinplot(
            data[key][feature_names], positions=positions, showextrema=False, widths=0.8
        )

        for violin in violins["bodies"]:
            violin.set_facecolor("none")
            violin.set_edgecolor(cmap(index))
            violin.set_linewidth(0.5)
            violin.set_alpha(1)

        means = data[key][feature_names].mean(axis=0)
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
