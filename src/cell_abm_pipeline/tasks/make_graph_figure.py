import matplotlib.figure as mpl
import matplotlib.pyplot as plt
import pandas as pd
from prefect import task


@task
def make_graph_figure(
    node_data: pd.DataFrame, edge_data: pd.DataFrame, colormap: str
) -> mpl.Figure:
    fig = plt.figure(figsize=(4, 4), constrained_layout=True)

    ax = fig.add_subplot()
    ax.set_box_aspect(1)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    for spine in ax.spines.values():
        spine.set_edgecolor("#dddddd")

    node_data = node_data.set_index("id")

    for edge in edge_data.to_dict("records"):
        x1, y1 = node_data.loc[edge["id1"]][["x", "y"]]
        x2, y2 = node_data.loc[edge["id2"]][["x", "y"]]

        ax.plot([x1, x2], [y1, y2], color="k", lw=0.5, zorder=1)

    ax.scatter(node_data["x"], node_data["y"], c=node_data["v"], cmap=colormap, s=20)

    return fig
