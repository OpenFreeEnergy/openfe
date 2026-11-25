# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from pathlib import Path

import click

from openfecli import OFECommandPlugin
from openfecli.utils import write


def update_task_db_main(
    task_db_path: Path,
    max_tries: int,
):
    """
    Parameters
    ----------
    task_db_path : pathlib.Path
        Path to a task.db

    max_tries: int
        Value to update "max_tries" column to, for applicable rows


    Example
    -------
    >>> openfe status --task-db task.db

    ┌─────────────────────────────┬──────────────────┬─────────────────────┬───────┬───────────┐
    │ task_id                     │ status           │ last_modified       │ tries │ max_tries │
    ├─────────────────────────────┼──────────────────┼─────────────────────┼───────┼───────────┤
    │ SetupUnit-d568ebe569b445c7… │ COMPLETED        │ 2026-08-14 11:19:16 │ 1     │ 3         │
    │ SetupUnit-17dc05e0d79747e7… │ COMPLETED        │ 2026-08-14 11:19:21 │ 1     │ 3         │
    │ SetupUnit-78321eda905c4c2c… │ COMPLETED        │ 2026-08-14 11:19:22 │ 1     │ 3         │
    │ SetupUnit-0238f65b55044b1e… │ COMPLETED        │ 2026-08-14 11:19:22 │ 1     │ 3         │
    │ MultiStateSimulationUnit-2… │ COMPLETED        │ 2026-08-14 11:25:10 │ 1     │ 3         │
    │ MultiStateSimulationUnit-1… │ COMPLETED        │ 2026-08-14 11:25:34 │ 1     │ 3         │
    │ MultiStateSimulationUnit-f… │ COMPLETED        │ 2026-08-14 11:26:22 │ 1     │ 3         │
    │ MultiStateSimulationUnit-c… │ TOO_MANY_RETRIES │ 2026-08-14 11:26:24 │ 3     │ 3         │
    │ MultiStateAnalysisUnit-a72… │ COMPLETED        │ 2026-08-14 11:26:51 │ 1     │ 3         │
    │ MultiStateAnalysisUnit-e44… │ COMPLETED        │ 2026-08-14 11:26:26 │ 1     │ 3         │
    │ MultiStateAnalysisUnit-7e9… │ COMPLETED        │ 2026-08-14 11:29:07 │ 1     │ 3         │
    │ MultiStateAnalysisUnit-72c… │ BLOCKED          │ NaT                 │ 0     │ 3         │
    └─────────────────────────────┴──────────────────┴─────────────────────┴───────┴───────────┘

    >>> openfe update-task-db task.db --max-tries=6
    ┌─────────────────────────────┬──────────────────┬─────────────────────┬───────┬───────────┐
    │ task_id                     │ status           │ last_modified       │ tries │ max_tries │
    ├─────────────────────────────┼──────────────────┼─────────────────────┼───────┼───────────┤
    │ SetupUnit-d568ebe569b445c7… │ COMPLETED        │ 2026-08-14 11:19:16 │ 1     │ 3         │
    │ SetupUnit-17dc05e0d79747e7… │ COMPLETED        │ 2026-08-14 11:19:21 │ 1     │ 3         │
    │ SetupUnit-78321eda905c4c2c… │ COMPLETED        │ 2026-08-14 11:19:22 │ 1     │ 3         │
    │ SetupUnit-0238f65b55044b1e… │ COMPLETED        │ 2026-08-14 11:19:22 │ 1     │ 3         │
    │ MultiStateSimulationUnit-2… │ COMPLETED        │ 2026-08-14 11:25:10 │ 1     │ 3         │
    │ MultiStateSimulationUnit-1… │ COMPLETED        │ 2026-08-14 11:25:34 │ 1     │ 3         │
    │ MultiStateSimulationUnit-f… │ COMPLETED        │ 2026-08-14 11:26:22 │ 1     │ 3         │
    │ MultiStateSimulationUnit-c… │ AVAILABLE        │ 2026-08-14 11:27:24 │ 3     │ 6         │
    │ MultiStateAnalysisUnit-a72… │ COMPLETED        │ 2026-08-14 11:26:51 │ 1     │ 3         │
    │ MultiStateAnalysisUnit-e44… │ COMPLETED        │ 2026-08-14 11:26:26 │ 1     │ 3         │
    │ MultiStateAnalysisUnit-7e9… │ COMPLETED        │ 2026-08-14 11:29:07 │ 1     │ 3         │
    │ MultiStateAnalysisUnit-72c… │ BLOCKED          │ NaT                 │ 0     │ 6         │
    └─────────────────────────────┴──────────────────┴─────────────────────┴───────┴───────────┘

    """
    write(f"Setting max_tries={max_tries} for all eligible tasks.")
    from exorcist import TaskStatusDB

    from openfe.orchestration import get_status_summary_df, update_max_tries

    task_db = TaskStatusDB.from_filename(task_db_path)

    update_max_tries(task_db, max_tries)
    write(
        "For a for a summary of the current task status db, use:\n\t"
        + f"openfe status --task-db {task_db_path} --summary",
    )


@click.command("update-task-db", short_help="Update the 'max_tries' column.")
@click.option(
    "--task-db",
    type=click.Path(
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
    help="Path to a TaskDB file.",
)
@click.option(
    "--max-tries",
    type=click.IntRange(min=1),
    required=True,
    help="The positive integer value to assign as 'max_tries' for the updated columns.",
)
# NOTE: this is named intentionally broad so that we can add "update_task_type" in a future version
def update_task_db(task_db: Path, max_tries: int):
    """Update the 'max_tries' column for applicable tasks.

    COMPLETED tasks and tasks with tries > the input 'max-tries' will not be updated.

    If a task with status=TOO_MANY_RETRIES has `max_tries` increased, its status will be updated to 'AVAILABLE'.

    .. warning:: Task-based execution is an experimental feature and subject to change in future releases of openfe.

    """
    msg = "WARNING! Task-based execution is an experimental feature and subject to change in a future release of openfe."
    click.secho(msg, err=False, fg="yellow")

    update_task_db_main(task_db_path=task_db, max_tries=max_tries)


PLUGIN = OFECommandPlugin(command=update_task_db, section="Execution", requires_ofe=(1, 13))
