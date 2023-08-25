import os

import pendulum
import requests
from prefect import flow, get_run_logger, settings, task
from prefect.deployments import Deployment

BASE_URL = "http://127.0.0.1:4200/api"

STORAGE_PATH = settings.PREFECT_LOCAL_STORAGE_PATH.value()

API_LIMIT = settings.PREFECT_API_DEFAULT_LIMIT.value()


@task
def get_expired_caches(base_url: str) -> list[str]:
    """
    Gets cache storage keys for expired caches.

    Parameters
    ----------
    base_url
        URL for API.

    Returns
    -------
    :
        List of expired cache storage keys.
    """

    offset = 0
    expired_caches: list[str] = []

    while True:
        response = requests.post(
            url=f"{base_url}/task_runs/filter",
            json={
                "offset": offset,
                "flow_runs": {"start_time": {"before_": pendulum.now().isoformat()}},
            },
            timeout=10,
        )

        task_runs_with_caches = [
            (
                task_run["state"]["data"]["storage_key"],
                task_run["state"]["state_details"]["cache_expiration"],
            )
            for task_run in response.json()
            if task_run["state"]["state_details"]["cache_expiration"] is not None
        ]

        expired_caches = expired_caches + [
            storage_key
            for storage_key, cache_expiration in task_runs_with_caches
            if pendulum.parse(cache_expiration).is_past()  # type: ignore[union-attr]
        ]

        if len(response.json()) == API_LIMIT:
            offset = offset + API_LIMIT
        else:
            break

    return list(set(expired_caches))


@task
def remove_expired_cache(cache_key: str) -> None:
    """
    Remove the given cache at the storage path.

    Parameters
    ----------
    cache_key : str
        Key for the cache at the storage path.
    """

    logger = get_run_logger()
    cache_file = f"{STORAGE_PATH}/{cache_key}"

    if os.path.isfile(cache_file):
        logger.info("Removing expired cache key [ %s ]", cache_key)
        os.remove(cache_file)


@flow(name="remove-expired_caches")
def remove_expired_caches(base_url: str) -> None:
    """
    Finds and removes all expired caches.

    Parameters
    ----------
    base_url
        URL for API.
    """
    expired_caches = get_expired_caches(base_url)

    for cache_key in expired_caches:
        remove_expired_cache.submit(cache_key)


if __name__ == "__main__":
    deployment = Deployment.build_from_flow(
        flow=remove_expired_caches,
        name="remove-expired-caches",
        parameters={"base_url": BASE_URL},
    )

    deployment.apply()
