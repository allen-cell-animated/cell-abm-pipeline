from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_object_storage(keys, storage, groups):
    fig, gridspec, indices = make_grid_figure(groups)

    storage["STORAGE_MIB"] = storage["STORAGE"] / 1024**2
    storage = storage[storage["KEY"].isin(keys)]

    for i, j, group in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(group)
        ax.set_ylabel("Object storage size (MiB)")
        ax.set_xlim([-0.5, len(keys) - 0.5])

        storage_group = storage[storage["GROUP"] == group]

        values = [storage_group[storage_group["KEY"] == key]["STORAGE_MIB"].values for key in keys]
        ax.boxplot(values, labels=keys, positions=range(0, len(keys)))
        ax.scatter(
            storage_group["KEY"],
            storage_group["STORAGE_MIB"],
            s=10,
            alpha=0.3,
            c="k",
            edgecolors="none",
        )

    return fig
