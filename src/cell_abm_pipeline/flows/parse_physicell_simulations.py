from dataclasses import dataclass, field
import tempfile

from io_collection.load import load_tar
from io_collection.save import save_dataframe

from container_collection.manifest import filter_manifest_files
from io_collection.load import load_dataframe
from prefect import flow

from simulariumio.physicell.dep.pyMCDS import pyMCDS


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
        # TODO verify this will load tar
        tar_file = load_tar(context.working_location, key) 
        working_dir = context.working_location
        # read PhysiCell output files
        output_files = working_dir.glob("*output*.xml")
        file_mapping = {}
        for output_file in output_files:
            index = int(output_file.name[output_file.name.index("output") + 6 :].split(".")[0])
            file_mapping[index] = output_file
        data = []
        for _, xml_file in sorted(file_mapping.items()):
            data.append(pyMCDS(xml_file.name, False, working_dir))
        # TODO shape data for analysis and save
        print(data)
        save_dataframe(context.working_location, key, data, index=False)
