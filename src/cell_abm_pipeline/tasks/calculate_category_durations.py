from itertools import groupby

import pandas as pd
from prefect import task


@task
def calculate_category_durations(data: pd.DataFrame, category: str, key: str) -> list[float]:
    durations: list[float] = []

    end = data["TICK"].max()
    key_data = data[data[category] == key]

    for _, group in key_data.groupby(["SEED", "ID"]):
        group.sort_values("TICK", inplace=True)
        items = [list(g) for _, g in groupby(enumerate(group["TICK"]), lambda x: x[0] - x[1])]
        durations = durations + [
            group.iloc[item[-1][0]]["time"] - group.iloc[item[0][0]]["time"]
            for item in items
            if item[-1][1] != end
        ]

    return durations
