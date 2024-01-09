import matplotlib as mpl
import matplotlib.pyplot as plt
from prefect import task


@task
def make_contour_figure(
    data: dict,
    index: int,
    view: str,
    regions: list[str],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    colors: dict[str, str],
) -> mpl.figure.Figure:

    fig = plt.figure(
        figsize=(3, 3 * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])), constrained_layout=True
    )

    ax = fig.add_subplot()

    if view == "top":
        ax.invert_yaxis()
        ax.set_xlim(xlim)
        ax.set_ylim((ylim[1], ylim[0]))
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_facecolor("#000")

    for region in regions:
        color = colors[region]

        if index not in data[region][view]:
            continue

        for contour in data[region][view][index]:
            ax.plot(contour[:, 0], contour[:, 1], linewidth=0.5, color=color)

    return fig
