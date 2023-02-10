from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure, remove_duplicate_legend


@task
def plot_graph_distances(keys, measures):
    fig, gridspec, indices = make_grid_figure(keys)

    for i, j, key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(key)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Distance")

        for _, group in measures[key].sort_values(by="TICK").groupby("SEED"):
            time = group["time"] / 24

            radius = group["RADIUS"]
            diameter = group["DIAMETER"]
            eccentricity = group["ECCENTRICITY"]
            shortest_path = group["SHORTEST_PATH"]

            ax.plot(time, eccentricity, c="#009", alpha=0.5, label="eccentricity")
            ax.plot(time, shortest_path, c="#900", alpha=0.5, label="shortest path")
            ax.plot(time, radius, c="#000", alpha=0.5, label="radius")
            ax.plot(time, diameter, c="#090", alpha=0.5, label="diameter")

        ax.legend()
        remove_duplicate_legend(ax)

    return fig
