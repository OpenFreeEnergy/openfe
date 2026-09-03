# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from pathlib import Path

import click

from openfecli import OFECommandPlugin
from openfecli.utils import configure_logger, print_duration, write


def _build_worker(warehouse_path: Path, task_db_path: Path):
    from openfe.orchestration import Worker
    from openfe.storage.warehouse import FileSystemWarehouse

    warehouse = FileSystemWarehouse(str(warehouse_path), exist_ok=True)
    return Worker(warehouse=warehouse, task_db_path=task_db_path)


def _write_failure_result_details(taskid: str, result) -> None:
    source_key = getattr(result, "source_key", None)
    exception = getattr(result, "exception", None)
    traceback_text = getattr(result, "traceback", None)

    write(f"Task '{taskid}' returned a failure result.")
    if source_key is not None:
        write(f"Failed unit source key: {source_key}")

    if isinstance(exception, tuple) and len(exception) == 2:
        exc_type, exc_args = exception
        write(f"Protocol unit exception: {exc_type}: {exc_args}")

    if isinstance(traceback_text, str) and traceback_text:
        write("Protocol unit traceback:")
        write(traceback_text)


def run_task_main(warehouse_path: Path, task_db_path: Path, scratch: Path):
    import logging
    import sys
    import traceback

    from openfe.utils import logging_control

    # avoid problems with output not showing if queueing system kills a job
    sys.stdout.reconfigure(line_buffering=True)

    stdout_handler = logging.StreamHandler(sys.stdout)

    configure_logger("gufekey", handler=stdout_handler)
    configure_logger("gufe", handler=stdout_handler)
    configure_logger("openfe", handler=stdout_handler)

    # silence the openmmtools.multistate API warning
    logging_control._silence_message(
        msg=[
            "The openmmtools.multistate API is experimental and may change in future releases",
        ],
        logger_names=[
            "openmmtools.multistate.multistatereporter",
            "openmmtools.multistate.multistateanalyzer",
            "openmmtools.multistate.multistatesampler",
        ],
    )
    # turn warnings into log message (don't show stack trace)
    logging.captureWarnings(True)
    if not task_db_path.is_file():
        raise click.ClickException(f"Task database not found at: {task_db_path}")

    scratch.mkdir(parents=True, exist_ok=True)

    worker = _build_worker(warehouse_path, task_db_path)

    try:
        write("Attempting to execute unit ...")
        execution = worker.execute_unit(scratch=scratch)
    except Exception as exc:
        write(traceback.format_exc())
        raise click.ClickException(f"Task execution failed: {exc}") from exc

    if execution is None:
        write("No available task in task graph.")
        return None

    taskid, result = execution
    if not result.ok():
        _write_failure_result_details(taskid, result)
        raise click.ClickException(f"Task '{taskid}' returned a failure result.")

    write(f"Completed task: {taskid}")
    return result


@click.command("run-task", short_help="Execute one available task from a filesystem warehouse")
@click.argument(
    "warehouse_path",
    type=click.Path(
        exists=True,
        readable=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
    # help="Path to a FileSystemWarehouse.",
)
@click.argument(
    "task_db_path",
    type=click.Path(
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
    # help="Path to a TaskDB instance.",
)
@click.option(
    "--scratch",
    "-s",
    default=Path("scratch/"),
    type=click.Path(
        writable=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
    help="Directory for scratch files. Defaults to 'scratch/' in the current working directory.",
)
@print_duration
def run_task(warehouse_path: Path, task_db_path: Path, scratch: Path):
    """
    Execute one available task from a warehouse task graph.

    The warehouse directory must contain a ``tasks.db`` task database and task
    payloads under ``tasks/`` created via OpenFE orchestration setup.
    """
    run_task_main(warehouse_path=warehouse_path, task_db_path=task_db_path, scratch=scratch)


PLUGIN = OFECommandPlugin(command=run_task, section="Execution", requires_ofe=(1, 13))
