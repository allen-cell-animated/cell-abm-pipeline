from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_cluster_fractions(keys, clusters):
    fig, gridspec, indices = make_grid_figure(keys)

    for i, j, key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(key)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Fraction in clusters")

        for _, group in clusters[key].sort_values(by="TICK").groupby("SEED"):
            time = group["time"] / 24

            num_clusters = group["NUM_CLUSTERS"]
            num_singles = group["NUM_SINGLES"]

            ax.plot(time, num_clusters / (num_clusters + num_singles), c="#000", alpha=0.5)

    return fig
