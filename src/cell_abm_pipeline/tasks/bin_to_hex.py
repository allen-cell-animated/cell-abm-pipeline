from math import sqrt
from typing import Optional, Union

import numpy as np
from prefect import task


@task
def bin_to_hex(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    values: np.ndarray,
    scale: float = 1,
    limits: Optional[tuple[float, float, float, float]] = None,
) -> dict[tuple[float, float], list[Union[int, float]]]:
    bins: dict[tuple[float, float], list[Union[int, float]]] = {}

    if limits is not None:
        x_min, x_max, y_min, y_max = limits
        x = (x_coordinates - x_min) / (x_max - x_min)
        y = (y_coordinates - y_min) / (y_max - y_min)
    else:
        x = x_coordinates
        y = y_coordinates

    for xi, yi, vi in zip(x, y, values):
        sxi = xi / scale
        syi = yi / scale

        cx1 = scale * round(sxi / sqrt(3)) * sqrt(3)
        cy1 = scale * round(syi)
        dist1 = sqrt((xi - cx1) ** 2 + (yi - cy1) ** 2)

        cx2 = scale * (round(sxi / sqrt(3) - 0.4999) + 0.5) * sqrt(3)
        cy2 = scale * (round(syi - 0.49999) + 0.5)
        dist2 = sqrt((xi - cx2) ** 2 + (yi - cy2) ** 2)

        if dist1 < dist2:
            cx, cy = (cx1, cy1)
        else:
            cx, cy = (cx2, cy2)

        if limits is not None:
            cx = (cx * (x_max - x_min)) + x_min
            cy = (cy * (y_max - y_min)) + y_min

        if (cx, cy) not in bins:
            bins[(cx, cy)] = []

        bins[(cx, cy)].append(vi)

    return bins
