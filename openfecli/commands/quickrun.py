# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import json
import pathlib

import click

from openfecli import OFECommandPlugin
from openfecli.utils import configure_logger, print_duration, write


def _format_exception(exception) -> str:
    """Takes the exception as stored by Gufe and reformats it."""
    return f"{exception[0]}: {exception[1][0]}"


@click.command("quickrun", short_help="Run a given transformation, saved as a JSON file")
@click.argument("transformation", type=click.File(mode="r"), required=True)
@click.option(
    "--work-dir", "-d", default=None,
    type=click.Path(dir_okay=True, file_okay=False, writable=True, path_type=pathlib.Path),
    help=(
        "Directory in which to store files in (defaults to current directory). "
        "If the directory does not exist, it will be created at runtime."
    ),
)  # fmt: skip
@click.option(
    "output", "-o", default=None,
    type=click.Path(dir_okay=False, file_okay=False, path_type=pathlib.Path),
    help="Filepath at which to create and write the JSON-formatted results.",
)  # fmt: skip
@print_duration
def quickrun(transformation, work_dir, output):
    """Run the transformation (edge) in the given JSON file.

    Simulation JSON files can be created with the
    :ref:`cli_plan-rbfe-network`
    or from Python a :class:`.Transformation` can be saved using its to_json
    method::

        transformation.to_json("filename.json")

    That will save a JSON file suitable to be input for this command.

    Running this command will execute the simulation defined in the JSON file,
    creating a directory for each individual task (``Unit``) in the workflow.
    For example, when running the OpenMM HREX Protocol a directory will be created
    for each repeat of the sampling process (by default 3).
    """
    import logging
    import os
    import sys

    from gufe.protocols.protocoldag import execute_DAG
    from gufe.tokenization import JSON_HANDLER
    from gufe.transformations.transformation import Transformation

    from openfe.utils import LogControl

    # avoid problems with output not showing if queueing system kills a job
    sys.stdout.reconfigure(line_buffering=True)

    stdout_handler = logging.StreamHandler(sys.stdout)

    configure_logger("gufekey", handler=stdout_handler)
    configure_logger("gufe", handler=stdout_handler)
    configure_logger("openfe", handler=stdout_handler)

    # silence the openmmtools.multistate API warning
    LogControl.silence_message(
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

    if work_dir is None:
        work_dir = pathlib.Path(os.getcwd())
    else:
        work_dir.mkdir(exist_ok=True, parents=True)

    write("Loading file...")
    trans = Transformation.from_json(transformation)

    if output is None:
        output = work_dir / (str(trans.key) + "_results.json")
    else:
        output.parent.mkdir(exist_ok=True, parents=True)

    write("Planning simulations for this edge...")
    dag = trans.create()
    write("Starting the simulations for this edge...")
    dagresult = execute_DAG(
        dag,
        shared_basedir=work_dir,
        scratch_basedir=work_dir,
        keep_shared=True,
        raise_error=False,
        n_retries=2,
    )
    write("Done with all simulations! Analyzing the results....")
    prot_result = trans.protocol.gather([dagresult])

    if dagresult.ok():
        estimate = prot_result.get_estimate()
        uncertainty = prot_result.get_uncertainty()
    else:
        estimate = uncertainty = None  # for output file

    out_dict = {
        "estimate": estimate,
        "uncertainty": uncertainty,
        "protocol_result": prot_result.to_dict(),
        "unit_results": {
            unit.key: unit.to_keyed_dict() for unit in dagresult.protocol_unit_results
        },
    }

    with open(output, mode="w") as outf:
        json.dump(out_dict, outf, cls=JSON_HANDLER.encoder)

    write(f"Here is the result:\n\tdG = {estimate} ± {uncertainty}\n")
    write("")

    if not dagresult.ok():
        # there can be only one, MacCleod
        failure = dagresult.protocol_unit_failures[-1]
        raise click.ClickException(
            f"The protocol unit '{failure.name}' failed with the error "
            f"message:\n{_format_exception(failure.exception)}\n\n"
            "Details provided in output."
        )


PLUGIN = OFECommandPlugin(command=quickrun, section="Quickrun Executor", requires_ofe=(0, 3))

if __name__ == "__main__":
    quickrun()
