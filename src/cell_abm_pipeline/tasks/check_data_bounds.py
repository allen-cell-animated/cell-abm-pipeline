import pandas as pd
from prefect import get_run_logger, task


@task
def check_data_bounds(data: pd.DataFrame, bounds: tuple[float, float], description: str) -> None:
    logger = get_run_logger()

    if data.max() > bounds[1]:
        logger.warning(
            "%s max [ %f ] greater than upper bound [ %f ]", description, data.max(), bounds[1]
        )

    if data.min() < bounds[0]:
        logger.warning(
            "%s min [ %f ] less than lower bound [ %f ]", description, data.min(), bounds[0]
        )
