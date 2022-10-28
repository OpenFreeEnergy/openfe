# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import click
from openfecli import OFECommandPlugin
from openfecli.parameters.output import ensure_file_does_not_exist
from openfecli.utils import write
import json
import pathlib


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
    from gufe.protocols.protocoldag import execute
    from gufe.tokenization import JSON_HANDLER

    write("Loading file...")
    # TODO: change this to `Transformation.load(transformation)`
    dct = json.load(transformation, cls=JSON_HANDLER.decoder)
    trans = gufe.Transformation.from_dict(dct)
    write("Planning simulations for this edge...")
    dag = trans.create()
    write("Running the simulations...")
    dagresult = execute(dag, shared=directory)
    write("Done! Analyzing the results....")

    if dagresult.ok():
        prot_result = trans.protocol.gather([dagresult])
        estimate = prot_result.get_estimate()
        uncertainty = prot_result.get_uncertainty()
    else:
        estimate = uncertainty = None  # for output file

    if output:
        with open(output, mode='w') as outf:
            out_dict = {
                'estimate': estimate,
                'uncertainty': uncertainty,
                'result': prot_result.to_dict()
            }
            json.dump(out_dict, outf, cls=JSON_HANDLER.encoder)

    write(f"Here is the result:\ndG = {estimate} ± {uncertainty}\n")
    write("Additional information:")
    for result in dagresult.protocol_unit_results:
        write(f"{result.name}:")
        write(result.outputs)

    if not dagresult.ok():
        # there can be only one, MacCleod
        failure = dagresult.protocol_unit_failures[-1]
        raise RuntimeError("A protocol unit failed with the error message:\n"
                           f"{failure.exception}\n"
                           "Details provided in output.")


PLUGIN = OFECommandPlugin(
    command=quickrun,
    section="Simulation",
    requires_ofe=(0, 3)
)
