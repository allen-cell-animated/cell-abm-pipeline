from dataclasses import dataclass

from prefect import flow


@dataclass
class ParametersConfig:
    name: str


@dataclass
class ContextConfig:
    name: str


@dataclass
class SeriesConfig:
    name: str


@flow(name="name-of-flow")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    print(context)
    print(series)
    print(parameters)
