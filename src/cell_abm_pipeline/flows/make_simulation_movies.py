"""
Workflow for making simulation movies.
"""

import ast
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from io_collection.keys import check_key, make_key
from io_collection.load import load_dataframe, load_tar
from io_collection.save import save_figure, save_gif
from prefect import flow

from cell_abm_pipeline.flows.plot_basic_metrics import PHASE_COLORS
from cell_abm_pipeline.tasks import make_graph_frame, make_projection_frame

MOVIES = [
    "projection",
    "graph",
]


@dataclass
class ParametersConfig:
    movies: list[str] = field(default_factory=lambda: MOVIES)

    regions: list[str] = field(default_factory=lambda: ["DEFAULT"])

    ds: float = 1.0

    dt: float = 1.0

    box: tuple[int, int, int] = (1, 1, 1)

    frame_spec: tuple[int, int, int] = (0, 1153, 48)

    scale: int = 100

    phase_colors: Optional[dict] = field(default_factory=lambda: PHASE_COLORS)


@dataclass
class ContextConfig:
    working_location: str


@dataclass
class SeriesConfig:
    name: str

    seeds: list[int]

    conditions: list[dict]


@flow(name="make-simulation-movies")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    if "projection" in parameters.movies:
        run_flow_make_projection(context, series, parameters)

    if "graph" in parameters.movies and "DEFAULT" in parameters.regions:
        run_flow_make_graph(context, series, parameters)


@flow(name="make-simulation-movies_make-projection")
def run_flow_make_projection(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    data_key = make_key(series.name, "data", "data.LOCATIONS")
    movie_key = make_key(series.name, "movies", "movies.PROJECTION")
    region_key = ":".join(sorted(parameters.regions))
    keys = [(condition["key"], seed) for condition in series.conditions for seed in series.seeds]

    all_locations = {}

    for key, seed in keys:
        series_key = f"{series.name}_{key}_{seed:04d}"
        locations_key = make_key(data_key, f"{series_key}.LOCATIONS.tar.xz")
        all_locations[(key, seed)] = load_tar(context.working_location, locations_key)

    frame_keys = []
    frames = list(np.arange(*parameters.frame_spec))

    for frame in frames:
        frame_key = make_key(movie_key, f"{series.name}_{region_key}_{frame:06d}.PROJECTION.png")
        frame_keys.append(frame_key)

        if check_key(context.working_location, frame_key):
            continue

        save_figure(
            context.working_location,
            frame_key,
            make_projection_frame(
                keys,
                all_locations,
                series.name,
                frame,
                parameters.scale,
                parameters.ds,
                parameters.dt,
                parameters.box,
                parameters.regions,
            ),
        )

    movie_key = make_key(movie_key, f"{series.name}_{region_key}.PROJECTION.gif")
    save_gif(context.working_location, movie_key, frame_keys)


@flow(name="make-simulation-movies_make-graph")
def run_flow_make_graph(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    analysis_key = make_key(series.name, "analysis", "analysis.NEIGHBORS")
    results_key = make_key(series.name, "results")
    movie_key = make_key(series.name, "movies", "movies.GRAPH")
    keys = [(condition["key"], seed) for condition in series.conditions for seed in series.seeds]

    all_neighbors = {}

    for key, seed in keys:
        series_key = f"{series.name}_{key}_{seed:04d}"

        results_key = make_key(series.name, "results", f"{series_key}.csv")
        results = load_dataframe(context.working_location, results_key)

        neighbors_key = make_key(analysis_key, f"{series_key}.NEIGHBORS.csv")
        neighbors = load_dataframe(
            context.working_location, neighbors_key, converters={"NEIGHBORS": ast.literal_eval}
        )

        combined_neighbors = neighbors.merge(results, on=["ID", "TICK"])
        all_neighbors[(key, seed)] = combined_neighbors

    frame_keys = []
    frames = list(np.arange(*parameters.frame_spec))

    for frame in frames:
        frame_key = make_key(movie_key, f"{series.name}_{frame:06d}.GRAPH.png")
        frame_keys.append(frame_key)

        if check_key(context.working_location, frame_key):
            continue

        save_figure(
            context.working_location,
            frame_key,
            make_graph_frame(
                keys,
                all_neighbors,
                frame,
                parameters.scale,
                parameters.ds,
                parameters.dt,
                parameters.box,
                parameters.phase_colors,
            ),
        )

    movie_key = make_key(movie_key, f"{series.name}.GRAPH.gif")
    save_gif(context.working_location, movie_key, frame_keys)
