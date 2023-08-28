"""
Workflow for summarizing files in the manifest.
"""

from dataclasses import dataclass, field
from fnmatch import fnmatch

from container_collection.manifest import summarize_manifest_files, update_manifest_contents
from io_collection.keys import get_keys, make_key
from io_collection.load import load_dataframe
from io_collection.save import save_dataframe, save_text
from prefect import flow


@dataclass
class ParametersConfig:
    """Parameter configuration for summarize manifest flow."""

    update_manifest: bool = True

    search_locations: list[str] = field(default_factory=lambda: [])

    include_filters: list[str] = field(default_factory=lambda: ["*"])

    exclude_filters: list[str] = field(default_factory=lambda: [])


@dataclass
class ContextConfig:
    """Context configuration for summarize manifest flow."""

    working_location: str

    manifest_location: str


@dataclass
class SeriesConfig:
    """Series configuration for summarize manifest flow."""

    name: str

    manifest_key: str

    seeds: list[int]

    conditions: list[dict]


@flow(name="summarize-manifest")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    """Main summarize manifest flow."""

    if parameters.update_manifest:
        location_keys = {}

        for location in parameters.search_locations:
            all_keys = get_keys(location, series.name)

            selected_keys = set()
            unselected_keys = set()

            # Filter files for matches to include filters.
            for include in parameters.include_filters:
                selected_keys.update([key for key in all_keys if fnmatch(key, include)])

            # Filter files for matches to exclude filters.
            for exclude in parameters.exclude_filters:
                unselected_keys.update([key for key in all_keys if fnmatch(key, exclude)])

            location_keys[location] = list(selected_keys - unselected_keys)

        manifest = update_manifest_contents(location_keys)
        save_dataframe(context.manifest_location, series.manifest_key, manifest, index=False)
    else:
        manifest = load_dataframe(context.manifest_location, series.manifest_key)

    summary = summarize_manifest_files(manifest, series.name, series.conditions, series.seeds)
    summary_key = make_key(series.name, "{{timestamp}}", f"{series.name}.SUMMARY.txt")
    save_text(context.working_location, summary_key, summary)

    print("\n" + summary)
