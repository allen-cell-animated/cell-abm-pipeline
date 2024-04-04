from itertools import groupby

import pandas as pd
from prefect import task


@task
def calculate_category_durations(
    data: pd.DataFrame, category: str, key: str, threshold: float = 0
) -> list[float]:
    durations: list[float] = []

    end = data["time"].max()
    key_data = data[data[category] == key]

    for _, group in key_data.groupby(["SEED", "ID"]):
        group.sort_values("time", inplace=True)
        items = [
            list(grouping)
            for valid, grouping in groupby(
                zip(group["time"], group["time"][1:]), lambda x: x[1] - x[0] < threshold
            )
            if valid
        ]
        durations = durations + [
            item[-1][1] - item[0][0] for item in items if item[0][0] != 0 and item[-1][1] != end
        ]

    return durations
