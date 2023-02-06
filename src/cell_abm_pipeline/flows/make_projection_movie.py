from dataclasses import dataclass
from typing import Optional

import numpy as np
from io_collection.keys import make_key
from io_collection.load import load_tar
from io_collection.save import save_figure
from prefect import flow

from cell_abm_pipeline.tasks import make_projection_frame, save_gif


@dataclass
class ParametersConfig:
    region: Optional[str] = None

    ds: float = 1.0

    dt: float = 1.0

    scale: float = 100

    box: tuple[int, int, int] = (1, 1, 1)

    frames: tuple[int, int, int] = (0, 1153, 48)


@dataclass
class ContextConfig:
    working_location: str


@dataclass
class SeriesConfig:
    name: str

    seeds: list[int]

    conditions: list[dict]


@flow(name="make_projection_movie")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    region_key = f"_{parameters.region}" if parameters.region is not None else ""
    keys = [(condition["key"], seed) for condition in series.conditions for seed in series.seeds]

    all_data = {}

    for key, seed in keys:
        series_key = f"{series.name}_{key}_{seed:04d}"
        data_key = make_key(series.name, "data", "data.LOCATIONS", f"{series_key}.LOCATIONS.tar.xz")
        all_data[(key, seed)] = load_tar(context.working_location, data_key)

    frame_keys = []
    for frame in np.arange(*parameters.frames):
        frame_key = make_key(
            series.name,
            "movies",
            "movies.PROJECTION",
            f"{series.name}{region_key}_{frame:06d}.PROJECTION.png",
        )
        frame_keys.append(frame_key)

        save_figure(
            context.working_location,
            frame_key,
            make_projection_frame(
                keys,
                all_data,
                series.name,
                frame,
                parameters.scale,
                parameters.ds,
                parameters.dt,
                parameters.box,
                parameters.region,
            ),
        )

    movie_key = make_key(
        series.name, "movies", "movies.PROJECTION", f"{series.name}{region_key}.PROJECTION.gif"
    )
    save_gif(context.working_location, movie_key, frame_keys)
