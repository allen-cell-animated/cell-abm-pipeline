from dataclasses import dataclass, field

from prefect import flow

FORMATS: list[str] = [
    "simularium",
]


@dataclass
class ParametersConfig:
    formats: list[str] = field(default_factory=lambda: FORMATS)


@dataclass
class ContextConfig:
    working_location: str


@dataclass
class SeriesConfig:
    name: str

    seeds: list[int]

    conditions: list[dict]


@flow(name="convert-physicell-format")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    if "simularium" in parameters.formats:
        run_flow_convert_to_simularium(context, series, parameters)


@flow(name="convert-physicell-format_convert-to-simularium")
def run_flow_convert_to_simularium(
    context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig
) -> None:
    # TODO: implement convert to simularium format
    print(context, series, parameters)
