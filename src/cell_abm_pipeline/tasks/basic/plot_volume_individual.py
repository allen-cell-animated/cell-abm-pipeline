from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_volume_individual(keys, data, region=None):
    fig, gridspec, indices = make_grid_figure(keys)
    value = f"volume.{region}" if region else "volume"

    for i, j, key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(key)
        ax.set_xlabel("Time (hrs)")
        ax.set_ylabel("Volume ($\mu m^3$)")

        key_data = data[key]

        for _, group in key_data.groupby(["SEED", "ID"]):
            group.sort_values("time", inplace=True)

            volume = group[value].values
            time = group["time"]

            ax.plot(time, volume, lw=0.5, alpha=0.5)

    return fig
