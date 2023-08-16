import matplotlib.figure as mpl
import matplotlib.pyplot as plt
from prefect import task


@task
def make_histogram_figure(keys: list[str], data: dict) -> mpl.Figure:
    fig = plt.figure(figsize=(4, 4), constrained_layout=True)

    ax = fig.add_subplot()
    ax.set_box_aspect(1)

    for key in keys:
        width = data["*"]["bandwidth"]
        edges = [item["x"] for item in data[key]] + [width]
        values = [item["y"] for item in data[key]]
        ax.stairs(values, edges)

    return fig
