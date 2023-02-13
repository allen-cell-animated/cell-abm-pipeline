import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_volume_locations(keys, data, tick=0, reference=None, region=None, padding=50):
    fig, gridspec, indices = make_grid_figure(keys)
    value = f"volume.{region}" if region else "volume"

    all_data = pd.concat(data.values())
    xlim = (all_data["CENTER_X"].min() - padding, all_data["CENTER_X"].max() + padding)
    ylim = (all_data["CENTER_Y"].min() - padding, all_data["CENTER_Y"].max() + padding)

    if reference is not None:
        reference_volume = reference[value].mean()
    else:
        reference_volume = all_data[all_data["TICK"] == tick][value].mean()

    for i, j, (key, seed) in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(f"{key} [{seed}]")
        ax.invert_yaxis()
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        key_seed_data = data[(key, seed)]
        tick_data = key_seed_data[key_seed_data["TICK"] == tick]

        x = tick_data["CENTER_X"]
        y = tick_data["CENTER_Y"]
        volumes = (tick_data[value] - reference_volume) / reference_volume

        sax = ax.scatter(x, y, c=volumes, s=20, cmap="coolwarm", vmin=-1, vmax=1)
        cbax = inset_axes(ax, width="3%", height="96%", loc="upper right")
        colorbar = fig.colorbar(sax, cax=cbax)
        colorbar.ax.yaxis.set_ticks_position("left")

    return fig
