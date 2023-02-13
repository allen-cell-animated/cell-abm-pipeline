import pandas as pd
from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_population_locations(keys, data, tick=0, padding=50):
    fig, gridspec, indices = make_grid_figure(keys)

    all_data = pd.concat(data.values())
    max_x = all_data["CENTER_X"].max()
    min_x = all_data["CENTER_X"].min()
    max_y = all_data["CENTER_Y"].max()
    min_y = all_data["CENTER_Y"].min()

    for i, j, (key, seed) in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(f"{key} [{seed}]")
        ax.invert_yaxis()
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_xlim((min_x - padding, max_x + padding))
        ax.set_ylim((min_y - padding, max_y + padding))

        key_seed_data = data[(key, seed)]
        tick_data = key_seed_data[key_seed_data["TICK"] == tick]

        x = tick_data["CENTER_X"]
        y = tick_data["CENTER_Y"]
        populations = tick_data["POPULATION"]

        ax.scatter(x, y, c=populations, s=20, cmap="tab10", vmin=1, vmax=11)

    return fig
