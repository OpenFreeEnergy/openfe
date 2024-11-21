# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import click
import json
import pathlib

from openfecli import OFECommandPlugin
from openfecli.parameters.output import ensure_file_does_not_exist
from openfecli.utils import write, print_duration, configure_logger


def _format_exception(exception) -> str:
    """Takes the exception as stored by Gufe and reformats it.
    """
    return f"{exception[0]}: {exception[1][0]}"



@click.command(
    'quickrun',
    short_help="Run a given transformation, saved as a JSON file"
)
@click.argument('transformation', type=click.File(mode='r'),
                required=True)
@click.option(
    '--work-dir', '-d', default=None,
    type=click.Path(dir_okay=True, file_okay=False, writable=True,
                    path_type=pathlib.Path),
    help=(
        "directory to store files in (defaults to current directory)"
    ),
)
@click.option(
    'output', '-o', default=None,
    type=click.Path(dir_okay=False, file_okay=True, writable=True,
                    path_type=pathlib.Path),
    help="output file (JSON format) for the final results",
    callback=ensure_file_does_not_exist,
)
@print_duration
def quickrun(transformation, work_dir, output):
    """Run the transformation (edge) in the given JSON file.

    Simulation JSON files can be created with the
    :ref:`cli_plan-rbfe-network`
    or from Python a :class:`.Transformation` can be saved using its dump
    method::

        transformation.dump("filename.json")

    That will save a JSON file suitable to be input for this command.

    Running this command will execute the simulation defined in the JSON file,
    creating a directory for each individual task (``Unit``) in the workflow.
    For example, when running the OpenMM HREX Protocol a directory will be created
    for each repeat of the sampling process (by default 3).
    """
    import gufe
    import os
    import sys
    from gufe.protocols.protocoldag import execute_DAG
    from gufe.tokenization import JSON_HANDLER
    from openfe.utils.logging_filter import MsgIncludesStringFilter
    import logging

    # avoid problems with output not showing if queueing system kills a job
    sys.stdout.reconfigure(line_buffering=True)

    stdout_handler = logging.StreamHandler(sys.stdout)

    configure_logger('gufekey', handler=stdout_handler)
    configure_logger('gufe', handler=stdout_handler)
    configure_logger('openfe', handler=stdout_handler)

    # silence the openmmtools.multistate API warning
    stfu = MsgIncludesStringFilter(
        "The openmmtools.multistate API is experimental and may change in "
        "future releases"
    )
    omm_multistate = "openmmtools.multistate"
    modules = ["multistatereporter", "multistateanalyzer",
               "multistatesampler"]
    for module in modules:
        ms_log = logging.getLogger(omm_multistate + "." + module)
        ms_log.addFilter(stfu)

    # turn warnings into log message (don't show stack trace)
    logging.captureWarnings(True)

    if work_dir is None:
        work_dir = pathlib.Path(os.getcwd())
    else:
        work_dir.mkdir(exist_ok=True, parents=True)

    write("Loading file...")
    # TODO: change this to `Transformation.load(transformation)`
    dct = json.load(transformation, cls=JSON_HANDLER.decoder)
    trans = gufe.Transformation.from_dict(dct)

    if output is None:
        output = work_dir / (str(trans.key) + '_results.json')

    try:
        # make sure the output file is writeable before running simulations
        with open(output, mode='w') as outf:
            outf.write("in-progress")  # TODO: do we just want to pass here?
    except FileNotFoundError as e:
        raise click.ClickException(f"Unable to write to {output}, please provide a valid path.")

    # TODO: if it's just a missing directory, should we try creating it?

    write("Planning simulations for this edge...")
    dag = trans.create()
    write("Starting the simulations for this edge...")
    dagresult = execute_DAG(dag,
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
        'estimate': estimate,
        'uncertainty': uncertainty,
        'protocol_result': prot_result.to_dict(),
        'unit_results': {
            unit.key: unit.to_keyed_dict()
            for unit in dagresult.protocol_unit_results
        }
    }


    with open(output, mode='w') as outf:
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


PLUGIN = OFECommandPlugin(
    command=quickrun,
    section="Quickrun Executor",
    requires_ofe=(0, 3)
)

if __name__ == "__main__":
    quickrun()
