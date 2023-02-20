import matplotlib.figure as mpl
import pandas as pd
from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_object_storage(keys: list[str], storage: pd.DataFrame, groups: list[str]) -> mpl.Figure:
    fig, gridspec, indices = make_grid_figure(groups)

    storage["STORAGE_MIB"] = storage["STORAGE"] / 1024**2
    storage = storage[storage["KEY"].isin(keys)]

    for i, j, group in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(group)
        ax.set_ylabel("Object storage size (MiB)")
        ax.set_xlim([-0.5, len(keys) - 0.5])

        storage_group = storage[storage["GROUP"] == group]

        for i, key in enumerate(keys):
            group = storage_group[storage_group["KEY"] == key]
            ax.boxplot(group["STORAGE_MIB"].values, labels=[key], positions=[i], widths=0.6)
            ax.scatter(
                group["KEY"],
                group["STORAGE_MIB"],
                s=10,
                alpha=0.3,
                c="k",
                edgecolors="none",
            )

    return fig
