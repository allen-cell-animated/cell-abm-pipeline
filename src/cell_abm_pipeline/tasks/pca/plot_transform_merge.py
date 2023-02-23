import matplotlib.figure as mpl
import pandas as pd
from matplotlib import cm
from prefect import task
from sklearn.decomposition import PCA

from cell_abm_pipeline.utilities.plot import make_single_figure


@task
def plot_transform_merge(
    keys: list[str],
    model: PCA,
    data: dict[str, pd.DataFrame],
    ref_data: pd.DataFrame,
) -> mpl.Figure:
    fig = make_single_figure()
    cmap = cm.get_cmap("tab10")

    columns = ref_data.filter(like="shcoeffs").columns
    ref_transform = model.transform(ref_data[columns].values)

    components = ref_transform.shape[1]
    positions = range(components)

    ax = fig.add_subplot()

    ax.plot([-1, components], [0, 0], color="#ddd", linewidth=0.5, zorder=-1)
    ax.set_xlim([-0.5, components - 0.5])
    ax.set_xticks(positions)
    ax.set_xticklabels([f"PC{i + 1}" for i in range(components)])

    ref_violins = ax.violinplot(ref_transform, positions=positions, showextrema=False, widths=0.8)

    for ref_violin in ref_violins["bodies"]:
        ref_violin.set_facecolor("#ccc")
        ref_violin.set_edgecolor("none")
        ref_violin.set_alpha(1)

    for index, key in enumerate(keys):
        key_transform = model.transform(data[key][columns].values)
        violins = ax.violinplot(key_transform, positions=positions, showextrema=False, widths=0.8)

        for violin in violins["bodies"]:
            violin.set_facecolor("none")
            violin.set_edgecolor(cmap(index))
            violin.set_linewidth(0.5)
            violin.set_alpha(1)

        means = key_transform.mean(axis=0)
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
