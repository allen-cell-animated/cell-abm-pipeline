"""
Workflow for summarizing simulation files.
"""

from dataclasses import dataclass, field

from container_collection.manifest import summarize_manifest_files, update_manifest_contents
from io_collection.keys import get_keys, make_key
from io_collection.load import load_dataframe
from io_collection.save import save_dataframe, save_text
from prefect import flow


@dataclass
class ParametersConfig:
    update_manifest: bool = True

    search_locations: list[str] = field(default_factory=lambda: [])


@dataclass
class ContextConfig:
    working_location: str

    manifest_location: str


@dataclass
class SeriesConfig:
    name: str

    manifest_key: str

    seeds: list[int]

    conditions: list[dict]


@flow(name="summarize-simulations")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    if parameters.update_manifest:
        location_keys = {
            location: get_keys(location, series.name) for location in parameters.search_locations
        }
        manifest = update_manifest_contents(location_keys)
        save_dataframe(context.manifest_location, series.manifest_key, manifest, index=False)
    else:
        manifest = load_dataframe(context.manifest_location, series.manifest_key)

    summary = summarize_manifest_files(manifest, series.name, series.conditions, series.seeds)
    summary_key = make_key(series.name, "{{timestamp}}", f"{series.name}.SUMMARY.txt")
    save_text(context.working_location, summary_key, summary)

    print(summary)
