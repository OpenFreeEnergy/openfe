# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pathlib

import click

from openfecli import OFECommandPlugin
from openfecli.utils import print_duration, write


def _build_worker(warehouse_path: pathlib.Path, db_path: pathlib.Path):
    from openfe.orchestration import Worker
    from openfe.storage.warehouse import FileSystemWarehouse

    warehouse = FileSystemWarehouse(str(warehouse_path))
    return Worker(warehouse=warehouse, task_db_path=db_path)


def worker_main(warehouse_path: pathlib.Path, scratch: pathlib.Path | None):
    db_path = warehouse_path / "tasks.db"
    if not db_path.is_file():
        raise click.ClickException(f"Task database not found at: {db_path}")

    if scratch is None:
        scratch = pathlib.Path.cwd()

    scratch.mkdir(parents=True, exist_ok=True)

    worker = _build_worker(warehouse_path, db_path)

    try:
        execution = worker.execute_unit(scratch=scratch)
    except Exception as exc:
        raise click.ClickException(f"Task execution failed: {exc}") from exc

    if execution is None:
        write("No available task in task graph.")
        return None

    taskid, result = execution
    if not result.ok():
        raise click.ClickException(f"Task '{taskid}' returned a failure result.")

    write(f"Completed task: {taskid}")
    return result


@click.command("worker", short_help="Execute one available task from a filesystem warehouse")
@click.argument(
    "warehouse_path",
    type=click.Path(
        exists=True,
        readable=True,
        file_okay=False,
        dir_okay=True,
        path_type=pathlib.Path,
    ),
)
@click.option(
    "--scratch",
    "-s",
    default=None,
    type=click.Path(
        writable=True,
        file_okay=False,
        dir_okay=True,
        path_type=pathlib.Path,
    ),
    help="Directory for scratch files. Defaults to current working directory.",
)
@print_duration
def worker(warehouse_path: pathlib.Path, scratch: pathlib.Path | None):
    """
    Execute one available task from a warehouse task graph.

    The warehouse directory must contain a ``tasks.db`` task database and task
    payloads under ``tasks/`` created via OpenFE orchestration setup.
    """
    worker_main(warehouse_path=warehouse_path, scratch=scratch)


PLUGIN = OFECommandPlugin(command=worker, section="Quickrun Executor", requires_ofe=(0, 3))
