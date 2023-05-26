from dataclasses import dataclass, field

from container_collection.manifest import filter_manifest_files
from io_collection.load import load_dataframe
from prefect import flow


@dataclass
class ParametersConfig:
    include_filters: list[str] = field(default_factory=lambda: ["*"])

    exclude_filters: list[str] = field(default_factory=lambda: [])


@dataclass
class ContextConfig:
    working_location: str

    manifest_location: str


@dataclass
class SeriesConfig:
    name: str

    manifest_key: str

    extensions: list[str]


@flow(name="parse-physicell-simulations")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    manifest = load_dataframe(context.manifest_location, series.manifest_key)
    filtered_files = filter_manifest_files(
        manifest, series.extensions, parameters.include_filters, parameters.exclude_filters
    )

    for key, files in filtered_files.items():
        print(key, files)
        # TODO: implement parse simulation results
