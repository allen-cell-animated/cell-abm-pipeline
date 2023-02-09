from matplotlib import cm
from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_ks_by_key(keys, stats):
    fig, gridspec, indices = make_grid_figure(keys)
    cmap = cm.get_cmap("tab20")

    stats_by_tick = stats[stats["SUBSET"] != "all_ticks"]

    for i, j, key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(key)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Kolmogorov–Smirnov statistic")

        key_stats = stats_by_tick[stats_by_tick["KEY"] == key]
        for index, (feature, group) in enumerate(key_stats.groupby("FEATURE")):
            ax.plot(
                group["TIME"] / 24,
                group["KS_STATISTIC"],
                marker="o",
                linewidth=0.5,
                markersize=4,
                label=feature,
                color=cmap(index),
            )

        ax.legend()

    return fig
