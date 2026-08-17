# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pathlib

import click

from openfecli import OFECommandPlugin
from openfecli.utils import rich_print_to_stdout


def status_main(
    task_db_path: pathlib.Path,
):
    from exorcist import TaskStatusDB

    from openfe.orchestration import get_task_df

    task_db = TaskStatusDB.from_filename(task_db_path)
    task_df = get_task_df(task_db)
    # TODO: add task_type back in once it's used
    rich_print_to_stdout(task_df.drop("task_type", axis=1))


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
    Execute one available task from a warehouse task graph.

    The warehouse directory must contain a ``tasks.db`` task database and task
    payloads under ``tasks/`` created via OpenFE orchestration setup.
    """
    status_main(task_db_path=task_db_path)


PLUGIN = OFECommandPlugin(command=status, section="Quickrun Executor", requires_ofe=(0, 3))
