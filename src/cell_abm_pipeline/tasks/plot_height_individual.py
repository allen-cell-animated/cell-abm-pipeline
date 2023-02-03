from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_height_individual(keys, results, region=None):
    fig, gridspec, indices = make_grid_figure(keys)
    value = f"height.{region}" if region else "height"

    for i, j, key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(key)
        ax.set_xlabel("Time (hrs)")
        ax.set_ylabel("Height ($\mu m$)")

        key_results = results[key]
        counter = 0

        for _, group in key_results.groupby(["SEED", "ID"]):
            group.sort_values("time", inplace=True)
            counter = counter + 1

            height = group[value].values
            time = group["time"]

            ax.plot(time, height, lw=0.5, alpha=0.5)

    return fig
