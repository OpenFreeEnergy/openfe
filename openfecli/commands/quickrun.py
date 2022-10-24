# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import click
from openfecli import OFECommandPlugin
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
    type=click.File(mode='w'),
    help="output file (JSON) for results"
)
def quickrun(transformation, directory, output):
    """Run the transformation (edge) in the given JSON file in serial.

    To save a transformation as JSON, use the following Python recipe (after
    creating the transformation):

    \b
        import json
        with open(filename, 'w') as f:
            json.dump(transformation.to_dict(), f)
    """
    import gufe
    from gufe.protocols.protocoldag import execute
    from gufe.tokenization import JSON_HANDLER

    write("Loading file...")
    dct = json.load(transformation)
    trans = gufe.Transformation.from_dict(dct)
    write("Planning simulations for this edge...")
    dag = trans.create()
    write("Running the simulations...")
    dagresult = execute(dag, shared=directory)
    write("Done! Analyzing the results....")

    prot_result = trans.protocol.gather([dagresult])
    estimate = prot_result.get_estimate()
    uncertainty = prot_result.get_uncertainty()

    if output:
        with output as outf:
            json.dump(prot_result.to_dict(), outf, cls=JSON_HANDLER.encoder)


    write(f"Here is the result:\ndG = {estimate} ± {uncertainty}\n")
    write("Additional information:")
    for result in dagresult.protocol_unit_results:
        write(f"{result.name}:")
        write(result.outputs)


PLUGIN = OFECommandPlugin(
    command=quickrun,
    section="Simulation",
    requires_ofe=(0, 3)
)
