"""
Demo workflow.
"""

from prefect import flow

from cell_abm_pipeline.__config__ import ContextConfig, ParametersConfig, SeriesConfig


@flow(name="demo")
def run_flow(context: ContextConfig, series: SeriesConfig, parameters: ParametersConfig) -> None:
    """Main demo flow."""

    print(context)
    print(series)
    print(parameters)
