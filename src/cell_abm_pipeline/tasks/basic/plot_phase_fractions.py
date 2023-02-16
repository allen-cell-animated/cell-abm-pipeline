import matplotlib.figure as mpl
import pandas as pd
from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_phase_fractions(
    keys: list[str], data: dict[str, pd.DataFrame], phases: list[str], phase_colors: dict[str, str]
) -> mpl.Figure:
    fig, gridspec, indices = make_grid_figure(keys)

    for i, j, key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(key)
        ax.set_xlabel("Time (hrs)")
        ax.set_ylabel("Fraction of cells")

        key_data = data[key]
        total_count = key_data.groupby(["SEED", "time"]).size()

        for phase in phases:
            color = phase_colors[phase]
            count = key_data[key_data["PHASE"] == phase].groupby(["SEED", "time"]).size()
            fraction = count / total_count

            mean = fraction.groupby("time").mean()
            std = fraction.groupby(["time"]).std()
            time = mean.index

            label = f"{phase} ({mean.mean()*100:.1f} %)"
            ax.plot(time, mean, color=color, label=label)
            ax.fill_between(time, mean - std, mean + std, alpha=0.5, color=color)

        ax.legend()

    return fig
