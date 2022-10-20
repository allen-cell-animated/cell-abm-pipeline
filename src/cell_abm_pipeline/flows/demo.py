from prefect import flow

from cell_abm_pipeline.__config__ import ParametersConfig, ContextConfig, SeriesConfig


@flow(name="demo")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    print(context)
    print(series)
    print(parameters)
