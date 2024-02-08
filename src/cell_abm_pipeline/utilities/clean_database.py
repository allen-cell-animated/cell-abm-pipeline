import os
import sqlite3

import pendulum
from prefect import flow, get_run_logger, settings
from prefect.deployments import Deployment

STORAGE_PATH = settings.PREFECT_LOCAL_STORAGE_PATH.value()

DATABASE_PATH = "/home/ec2-user/.prefect/orion.db"

RETENTION_PERIOD = 48

DATETIME_FORMAT = "YYYY-MM-DD HH:mm:ss"


def get_delete_command(target: str, source: str) -> str:
    """
    Compiles the delete command for target table based on ids in source table.

    Parameters
    ----------
    target
        Name of table to delete from.
    source
        Name of table used to identify ids to delete.
    """

    return f"DELETE FROM {target} WHERE NOT EXISTS (SELECT NULL FROM {source} WHERE {source}.id = {target}.{source}_id)"


@flow(name="clean-database")
def clean_database(database_path: str, retention_period: int) -> None:
    """
    Finds and removes outdated flows, tasks, logs, and artifacts from database.

    Parameters
    ----------
    database_path
        Path to database file.
    retention_period
        Retention period (in hours).
    """

    logger = get_run_logger()

    conn = sqlite3.connect(database_path, isolation_level=None)
    cur = conn.cursor()

    now = pendulum.now().in_tz("UTC")
    current_timestamp = now.format(DATETIME_FORMAT)
    retention_timestamp = now.subtract(hours=retention_period).format(DATETIME_FORMAT)

    # Get storage keys for expired caches.
    storage_keys = cur.execute(
        f"SELECT json_extract(data, '$.storage_key') FROM artifact WHERE task_run_id IN (SELECT task_run_id FROM task_run_state WHERE id IN (SELECT task_run_state_id FROM task_run_state_cache WHERE cache_expiration <= '{current_timestamp}'))"
    ).fetchall()

    # Remove expired caches.
    for storage_key in storage_keys:
        storage_file = f"{STORAGE_PATH}/{storage_key[0]}"

        if os.path.isfile(storage_file):
            logger.info("Removing expired cache key [ %s ]", storage_key[0])
            os.remove(storage_file)

    # Remove flows, tasks, logs, and artifacts outside retention period.
    cur.execute(f"DELETE FROM flow_run WHERE end_time <= '{retention_timestamp}'")
    cur.execute(get_delete_command("flow_run_state", "flow_run"))
    cur.execute(get_delete_command("task_run", "flow_run"))
    cur.execute(get_delete_command("task_run_state", "task_run"))
    cur.execute(get_delete_command("task_run_state_cache", "task_run_state"))
    cur.execute(get_delete_command("log", "flow_run"))
    cur.execute(get_delete_command("artifact", "flow_run"))
    cur.execute(f"DELETE FROM task_run_state_cache WHERE cache_expiration <= '{pendulum.now()}'")
    cur.execute("VACUUM")


if __name__ == "__main__":
    deployment = Deployment.build_from_flow(
        flow=clean_database,
        name="clean-database",
        parameters={
            "database_path": DATABASE_PATH,
            "retention_period": RETENTION_PERIOD,
        },
    )

    deployment.apply()
