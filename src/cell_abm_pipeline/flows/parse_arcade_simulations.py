"""
Workflow for parsing ARCADE simulations into tidy data.
"""

from dataclasses import dataclass, field

from arcade_collection.output import merge_parsed_results, parse_cells_file, parse_locations_file
from container_collection.manifest import filter_manifest_files
from io_collection.keys import make_key
from io_collection.load import load_dataframe, load_tar
from io_collection.save import save_dataframe
from prefect import flow


@dataclass
class ParametersConfig:
    """Parameter configuration for parse arcade simulations flow."""

    regions: list[str] = field(default_factory=lambda: [])

    include_filters: list[str] = field(default_factory=lambda: ["*"])

    exclude_filters: list[str] = field(default_factory=lambda: [])


@dataclass
class ContextConfig:
    """Context configuration for parse arcade simulations flow."""

    working_location: str

    manifest_location: str


@dataclass
class SeriesConfig:
    """Series configuration for parse arcade simulations flow."""

    name: str

    manifest_key: str

    extensions: list[str]


@flow(name="parse-arcade-simulations")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    manifest = load_dataframe(context.manifest_location, series.manifest_key)
    filtered_files = filter_manifest_files(
        manifest, series.extensions, parameters.include_filters, parameters.exclude_filters
    )

    for key, files in filtered_files.items():
        cells_tar = load_tar(**files["CELLS.tar.xz"])
        cells = parse_cells_file(cells_tar, parameters.regions)

        locs_tar = load_tar(**files["LOCATIONS.tar.xz"])
        locs = parse_locations_file(locs_tar, parameters.regions)

        results = merge_parsed_results(cells, locs)
        results_key = make_key(series.name, "{{timestamp}}", "results", f"{key}.csv")
        save_dataframe(context.working_location, results_key, results, index=False)
