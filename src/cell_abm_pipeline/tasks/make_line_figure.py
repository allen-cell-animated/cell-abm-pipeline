from typing import Optional

import matplotlib.figure as mpl
import matplotlib.pyplot as plt
from prefect import task


@task
def make_line_figure(
    data: list[dict], color_map: Optional[dict] = None, color_key: str = ""
) -> mpl.Figure:
    fig = plt.figure(figsize=(4, 4), constrained_layout=True)

    ax = fig.add_subplot()
    ax.set_box_aspect(1)

    for item in data:
        color = "#000000" if color_map is None else color_map[item[color_key]]
        ax.plot(item["time"], item["value"], color=color, linewidth=0.5)

    return fig
