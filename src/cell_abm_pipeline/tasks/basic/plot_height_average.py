from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_height_average(keys, data, reference=None, region=None):
    fig, gridspec, indices = make_grid_figure(keys)
    value = f"height.{region}" if region else "height"

    for i, j, key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(key)
        ax.set_xlabel("Time (hrs)")
        ax.set_ylabel("Average height ($\mu m$)")

        height = data[key].groupby(["SEED", "time"])[value].mean()
        mean = height.groupby(["time"]).mean()
        std = height.groupby(["time"]).std()
        time = mean.index

        if reference is not None:
            ref_height_mean = reference[value].mean()
            ref_height_std = reference[value].std()
            ref_label = f"reference ({ref_height_mean:.1f} $\pm$ {ref_height_std:.1f} $\mu m$)"
            ax.plot(time, [reference[value].mean()] * len(time), c="#555", lw=0.5, label=ref_label)

        label = f"simulated ({mean.mean():.1f} $\pm$ {std.mean():.1f} $\mu m$)"
        ax.plot(time, mean, c="#000", label=label)
        ax.fill_between(time, mean - std, mean + std, facecolor="#bbb")

        ax.legend()

    return fig
