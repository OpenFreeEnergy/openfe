# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from pathlib import Path

import click

from openfecli import OFECommandPlugin
from openfecli.utils import rich_print_to_stdout


def status_main(
    task_db_path: Path,
    count: bool,
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

    if count:
        rich_print_counts(task_df)
    else:
        task_df["last_modified"] = task_df["last_modified"].dt.floor("s")
        rich_print_to_stdout(task_df)


def rich_print_counts(task_df):
    from exorcist import TaskStatus
    from rich.console import Console
    from rich.table import Table

    val_counts = task_df.status.value_counts()

    table = Table()
    table.add_column("status", justify="left", no_wrap=True)
    table.add_column("count", justify="right", no_wrap=True)

    for status_type in TaskStatus:
        status_name = status_type.name
        table.add_row(str(status_name), str(val_counts.get(status_name, 0)))
    console = Console()
    console.print(table)

    pass


@click.command("status", short_help="Output the status of the task database as a table.")
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
    "--count",
    "-c",
    flag_value=True,
    default=False,
)
def status(task_db: Path, count: bool):
    """
    Show the status of a task.db as a table.


    """
    # TODO: add loading bar
    status_main(task_db_path=task_db, count=count)


PLUGIN = OFECommandPlugin(command=status, section="Execution", requires_ofe=(1, 13))
