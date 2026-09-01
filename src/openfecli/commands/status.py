# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pathlib

import click

from openfecli import OFECommandPlugin
from openfecli.utils import rich_print_to_stdout


def status_main(
    task_db_path: pathlib.Path,
):
    """
    Parameters
    ----------
    task_db_path : pathlib.Path
        Path to a task.db

    Example
    -------
    > openfe status task.db

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

    """

    from exorcist import TaskStatusDB

    from openfe.orchestration.exorcist_utils import get_task_df

    # # TODO: rewrite this using just sql and rich table?
    task_db = TaskStatusDB.from_filename(task_db_path)
    task_df = get_task_df(task_db)
    task_df["last_modified"] = task_df["last_modified"].dt.floor("s")
    rich_print_to_stdout(task_df)


@click.command("status", short_help="Output the status of the task database as a table.")
@click.argument(
    "task_db_path",
    type=click.Path(
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=False,
        path_type=pathlib.Path,
    ),
)
def status(task_db_path: pathlib.Path):
    """
    Show the status of a task.db as a table.


    """
    status_main(task_db_path=task_db_path)


PLUGIN = OFECommandPlugin(command=status, section="Execution", requires_ofe=(1, 13))
