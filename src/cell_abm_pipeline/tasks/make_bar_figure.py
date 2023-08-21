import matplotlib.figure as mpl
import matplotlib.pyplot as plt
import numpy as np
from prefect import task


@task
def make_bar_figure(keys: list[str], data: dict) -> mpl.Figure:
    fig = plt.figure(figsize=(4, 4), constrained_layout=True)

    ax = fig.add_subplot()
    ax.set_box_aspect(1)

    width = 0.9 / len(keys)
    offset = (width * (len(keys) - 1)) / 2
    labels = list(data[keys[0]].keys())

    for index, key in enumerate(keys):
        positions = np.arange(len(labels)) + index * width
        means = [data[key][label]["mean"] for label in labels]
        stds = [data[key][label]["std"] for label in labels]
        ax.bar(positions, means, yerr=stds, width=width)

    ax.set_xticks(np.arange(len(labels)) + offset, labels, rotation=90)

    return fig
