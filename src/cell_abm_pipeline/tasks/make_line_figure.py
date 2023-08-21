import matplotlib.figure as mpl
import matplotlib.pyplot as plt
from prefect import task


@task
def make_line_figure(data: list[dict]) -> mpl.Figure:
    fig = plt.figure(figsize=(4, 4), constrained_layout=True)

    ax = fig.add_subplot()
    ax.set_box_aspect(1)

    for item in data:
        color = item["color"] if "color" in item else "#000000"
        ax.plot(item["x"], item["y"], color=color, linewidth=0.5)

    return fig
