from math import sqrt
from typing import Union

import numpy as np
from prefect import task


@task
def bin_to_hex(
    x: list[Union[int, float]],
    y: list[Union[int, float]],
    v: list[Union[int, float]],
    scale: float = 1,
    rescale: bool = False,
) -> dict[tuple[float, float], list[Union[int, float]]]:
    bins: dict[tuple[float, float], list[Union[int, float]]] = {}

    if rescale:
        x_scale = np.max(np.abs(x))
        y_scale = np.max(np.abs(y))
    else:
        x_scale = 1
        y_scale = 1

    for xi, yi, vi in zip(x / x_scale, y / y_scale, v):
        sx = xi / scale
        sy = yi / scale

        cx1 = scale * round(sx / sqrt(3)) * sqrt(3)
        cy1 = scale * round(sy)
        dist1 = sqrt((xi - cx1) ** 2 + (yi - cy1) ** 2)

        cx2 = scale * (round(sx / sqrt(3) - 0.4999) + 0.5) * sqrt(3)
        cy2 = scale * (round(sy - 0.49999) + 0.5)
        dist2 = sqrt((xi - cx2) ** 2 + (yi - cy2) ** 2)

        if dist1 < dist2:
            center = (cx1 * x_scale, cy1 * y_scale)
        else:
            center = (cx2 * x_scale, cy2 * y_scale)

        if center not in bins:
            bins[center] = []

        bins[center].append(vi)

    return bins
