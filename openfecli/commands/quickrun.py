# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import click
from openfecli import OFECommandPlugin
from openfecli.parameters.output import ensure_file_does_not_exist
from openfecli.utils import write
import json
import pathlib


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
    'directory', '-d', default=None,
    type=click.Path(dir_okay=True, file_okay=False, writable=True,
                    path_type=pathlib.Path),
    help=(
        "directory to store files in (defaults to temporary directory)"
    ),
)
@click.option(
    'output', '-o', default=None,
    type=click.Path(dir_okay=False, file_okay=True, writable=True,
                    path_type=pathlib.Path),
    help="output file (JSON format) for the final results",
    callback=ensure_file_does_not_exist,
)
def quickrun(transformation, directory, output):
    """Run the transformation (edge) in the given JSON file in serial.

    To save a transformation as JSON, create the transformation and then
    save it with transformation.dump(filename).
    """
    import gufe
    from gufe.protocols.protocoldag import execute_DAG
    from gufe.tokenization import JSON_HANDLER

    write("Loading file...")
    # TODO: change this to `Transformation.load(transformation)`
    dct = json.load(transformation, cls=JSON_HANDLER.decoder)
    trans = gufe.Transformation.from_dict(dct)
    write("Planning simulations for this edge...")
    dag = trans.create()
    write("Running the simulations...")
    dagresult = execute_DAG(dag, shared=directory, raise_error=False)
    write("Done! Analyzing the results....")
    prot_result = trans.protocol.gather([dagresult])

    if dagresult.ok():
        estimate = prot_result.get_estimate()
        uncertainty = prot_result.get_uncertainty()
    else:
        estimate = uncertainty = None  # for output file

    # TODO: remove this ugly hack on next release
    #       strip out Settings objects in each unit_result inputs dict
    for _, dd in out_dict['unit_results'].items():
        dd['inputs'].pop('settings')

    out_dict = {
        'estimate': estimate,
        'uncertainty': uncertainty,
        'protocol_result': prot_result.to_dict(),
        'unit_results': {
            unit.key: unit.to_keyed_dict()
            for unit in dagresult.protocol_unit_results
        }
    }

    if output:
        with open(output, mode='w') as outf:
            json.dump(out_dict, outf, cls=JSON_HANDLER.encoder)

    write(f"Here is the result:\ndG = {estimate} ± {uncertainty}\n")
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
