# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import click
import json
import pathlib

from openfecli import OFECommandPlugin
from openfecli.parameters.output import ensure_file_does_not_exist
from openfecli.utils import write, print_duration


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
    """Run the transformation (edge) in the given JSON file in serial.

    A transformation can be saved as JSON using from Python using its dump
    method::

        transformation.dump("filename.json")

    That will save a JSON file suitable to be input for this command.
    """
    import gufe
    import os
    from gufe.protocols.protocoldag import execute_DAG
    from gufe.tokenization import JSON_HANDLER

    if work_dir is None:
        work_dir = pathlib.Path(os.getcwd())
    else:
        work_dir.mkdir(exist_ok=True, parents=True)

    write("Loading file...")
    # TODO: change this to `Transformation.load(transformation)`
    dct = json.load(transformation, cls=JSON_HANDLER.decoder)
    trans = gufe.Transformation.from_dict(dct)
    write("Planning simulations for this edge...")
    dag = trans.create()
    write("Running the simulations...")
    dagresult = execute_DAG(dag,
                            shared_basedir=work_dir,
                            scratch_basedir=work_dir,
                            keep_shared=True,
                            raise_error=False)
    write("Done! Analyzing the results....")
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

    if output is None:
        output = work_dir / (str(trans.key) + '_results.json')

    with open(output, mode='w') as outf:
        json.dump(out_dict, outf, cls=JSON_HANDLER.encoder)

    write(f"Here is the result:\n\tdG = {estimate} ± {uncertainty}\n")

    write("Additional information:")
    for result in dagresult.protocol_unit_results:
        write(f"{result.name}:")
        write(result.outputs)

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
    section="Simulation",
    requires_ofe=(0, 3)
)
