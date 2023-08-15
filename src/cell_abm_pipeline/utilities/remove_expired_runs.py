import pendulum
import requests
from prefect import flow, get_run_logger, settings, task
from prefect.deployments import Deployment

BASE_URL = "http://127.0.0.1:4200/api"

RETENTION_PERIOD = 48

API_LIMIT = settings.PREFECT_API_DEFAULT_LIMIT.value()


@task
def get_expired_runs(base_url: str, retention_period: int) -> list[str]:
    """
    Gets flow run ids for expired flow runs.

    Parameters
    ----------
    base_url
        URL for API.
    retention_period : int
        Retention period (in hours).

    Returns
    -------
    :
        List of expired run ids.
    """

    offset = 0
    retention_timestamp = pendulum.now().subtract(hours=retention_period).start_of("hour")
    expired_runs: list[str] = []

    while True:
        response = requests.post(
            url=f"{base_url}/flow_runs/filter",
            json={
                "offset": offset,
                "flow_runs": {"start_time": {"before_": retention_timestamp.isoformat()}},
            },
            timeout=10,
        )

        expired_runs = expired_runs + [flow_run["id"] for flow_run in response.json()]

        if len(response.json()) == API_LIMIT:
            offset = offset + API_LIMIT
        else:
            break

    return expired_runs


@task
def remove_expired_run(base_url: str, run_id: str) -> None:
    """
    Remove the given run.

    Parameters
    ----------
    base_url
        URL for API.
    run_id
        Run id.
    """

    logger = get_run_logger()
    logger.info("Removing expired flow run [ %s ]", run_id)
    requests.delete(f"{base_url}/flow_runs/{run_id}", timeout=10)


@flow(name="remove-expired-runs")
def remove_expired_runs(base_url: str, retention_period: int) -> None:
    """
    Finds and removes all expired runs outside retention perion.

    Parameters
    ----------
    base_url
        URL for API.
    retention_period : int
        Retention period (in hours).
    """

    expired_runs = get_expired_runs(base_url, retention_period)

    for run_id in expired_runs:
        remove_expired_run.submit(base_url, run_id)


if __name__ == "__main__":
    deployment = Deployment.build_from_flow(
        flow=remove_expired_runs,
        name="remove-expired-runs",
        parameters={
            "base_url": BASE_URL,
            "retention_period": RETENTION_PERIOD,
        },
    )

    deployment.apply()
