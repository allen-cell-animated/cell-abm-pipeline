import matplotlib.figure as mpl
import matplotlib.pyplot as plt
import numpy as np
from prefect import task


@task
def make_range_figure(data: dict) -> mpl.Figure:
    fig = plt.figure(figsize=(4, 4), constrained_layout=True)

    ax = fig.add_subplot()
    ax.set_box_aspect(1)
    ax.set_xlabel("Time (hours)")

    time = data["time"]
    mean = np.array([np.nan if d == "nan" else d for d in data["mean"]])
    std = np.array([np.nan if d == "nan" else d for d in data["std"]])
    mins = np.array([np.nan if d == "nan" else d for d in data["min"]])
    maxs = np.array([np.nan if d == "nan" else d for d in data["max"]])

    ax.fill_between(time, mins, maxs, color="#eee")
    ax.plot(time, mean + std, color="#000", linestyle="dashed", linewidth=0.5)
    ax.plot(time, mean - std, color="#000", linestyle="dashed", linewidth=0.5)
    ax.plot(time, mean, color="#000", linewidth=1)

    return fig
