from matplotlib import cm
from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_ks_by_feature(keys, stats, ref_stats=None):
    fig, gridspec, indices = make_grid_figure(stats["FEATURE"].unique())
    cmap = cm.get_cmap("tab20")

    stats_by_tick = stats[stats["SUBSET"] != "all_ticks"]

    for i, j, feature_key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(feature_key)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Kolmogorov–Smirnov statistic")

        feature_stats = stats_by_tick[stats_by_tick["FEATURE"] == feature_key]

        for index, (key, group) in enumerate(feature_stats.groupby("KEY")):
            if key not in keys:
                continue

            ax.plot(
                group["TIME"] / 24,
                group["KS_STATISTIC"],
                marker="o",
                linewidth=0.5,
                markersize=4,
                label=key,
                color=cmap(index),
            )

        if ref_stats is not None:
            ref_stats_by_tick = ref_stats[ref_stats["SUBSET"] != "all_ticks"]
            ref_group = ref_stats_by_tick[ref_stats_by_tick["FEATURE"] == feature_key]

            ax.plot(
                ref_group["TIME"] / 24,
                ref_group["KS_STATISTIC"],
                marker="o",
                markersize=2,
                label="reference",
                lw=0.5,
                color="#888",
            )

        ax.legend()

    return fig
