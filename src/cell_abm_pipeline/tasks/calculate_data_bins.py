import numpy as np
from prefect import task


@task
def calculate_data_bins(
    data: np.ndarray, bounds: tuple[float, float], bandwidth: float
) -> list[dict]:
    if len(data) == 0:
        return []

    total = len(data) * bandwidth
    num_bins = int((bounds[1] - bounds[0]) / bandwidth)
    lower = bounds[0] - 3 * bandwidth / 2
    upper = bounds[1] + 3 * bandwidth / 2

    bins = np.linspace(lower, upper, num_bins + 4).tolist()
    counts, _ = np.histogram(data, bins)

    return [
        {"n": count, "x": x0, "y": count / total, "m": (x0 + x1) / 2}
        for count, x0, x1 in zip(counts.tolist(), bins[:-1], bins[1:])
    ]
