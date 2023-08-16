import matplotlib.figure as mpl
import matplotlib.pyplot as plt
import pandas as pd
from prefect import task


@task
def make_bar_figure(keys: list[str], data: pd.DataFrame) -> mpl.Figure:
    fig = plt.figure(figsize=(4, 4), constrained_layout=True)

    ax = fig.add_subplot()
    ax.set_box_aspect(1)

    means = data.groupby("key")["count"].mean()
    stds = data.groupby("key")["count"].std(ddof=1)

    values = [means[key] for key in keys]
    errors = [stds[key] for key in keys]

    ax.bar(keys, values, yerr=errors)

    return fig
