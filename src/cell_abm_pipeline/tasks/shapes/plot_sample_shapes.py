from typing import Optional

import matplotlib.figure as mpl
import numpy as np
from abm_shape_collection.make_voxels_array import make_voxels_array
from arcade_collection.output.get_location_voxels import get_location_voxels
from numpy.random import SeedSequence, default_rng
from prefect import get_run_logger, task

from cell_abm_pipeline.utilities.plot import make_grid_figure


@task
def plot_sample_shapes(
    locations: list[dict],
    box: tuple[int, int, int],
    region: Optional[str] = None,
    sample_size: int = 1,
    random_seed: int = 0,
) -> mpl.Figure:
    logger = get_run_logger()

    region = None if region == "DEFAULT" else region
    rng = default_rng(SeedSequence(random_seed))

    valid_locations = [loc for loc in locations if len(get_location_voxels(loc, region)) > 0]

    keys = list(
        rng.choice(len(valid_locations), min(len(valid_locations), sample_size), replace=False)
    )

    fig, gridspec, indices = make_grid_figure(keys)
    height, width, length = box

    for i, j, index in indices:
        ax = fig.add_subplot(gridspec[i, j])
        ax.set_title(f"{valid_locations[index]['id']}")
        ax.invert_yaxis()
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_xlim([0, length - 1])
        ax.set_ylim([width - 1, 0])
        ax.set(aspect=1)

        ax_horz = ax.inset_axes([0, 1.005, 1, height / width], sharex=ax)
        ax_horz.set_ylim([0, height - 1])
        ax_horz.get_yaxis().set_ticks([])

        ax_vert = ax.inset_axes([1.005, 0, height / length, 1], sharey=ax)
        ax_vert.set_xlim([0, height - 1])
        ax_vert.get_xaxis().set_ticks([])

        array = make_voxels_array(get_location_voxels(valid_locations[index], region))

        for dim, shape, bound in zip(["z", "y", "x"], array.shape, box):
            if shape > bound:
                logger.warning("Insufficient %s padding for %d with box size %d", dim, shape, bound)

        pad_z, pad_y, pad_x = (np.array(box) - array.shape).clip(0)
        padding = (
            (0, pad_z),
            (int(pad_y / 2), pad_y - int(pad_y / 2)),
            (int(pad_x / 2), pad_x - int(pad_x / 2)),
        )

        padded_array = np.pad(array, padding)

        ax.imshow(padded_array.mean(axis=0), cmap="bone", interpolation="none")
        ax_horz.imshow(padded_array.mean(axis=1), cmap="bone", interpolation="none")
        ax_vert.imshow(padded_array.mean(axis=2).transpose(), cmap="bone", interpolation="none")

    return fig
