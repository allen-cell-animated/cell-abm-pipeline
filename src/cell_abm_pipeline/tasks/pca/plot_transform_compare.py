from prefect import task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_transform_compare(keys, component, model, data, ref_data):
    fig, gridspec, indices = make_grid_figure(keys)

    ref_transform = model.transform(ref_data.filter(like="shcoeffs").values)

    for i, j, key in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(f"{key}")
        ax.set_xlabel(f"PC {component + 1}")

        key_transform = model.transform(data[key].filter(like="shcoeffs").values)

        ax.hist(
            ref_transform[:, component],
            bins=20,
            density=True,
            alpha=0.3,
            color="black",
            label="reference",
        )
        ax.hist(
            key_transform[:, component],
            bins=20,
            color="black",
            density=True,
            histtype="step",
            label="data",
        )

        ax.legend()

    return fig
