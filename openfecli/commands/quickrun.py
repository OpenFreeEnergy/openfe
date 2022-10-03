# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import click
from openfecli import OFECommandPlugin
import json


@click.command(
    'quickrun',
    short_help="Run a given transformation, saved as a JSON file"
)
@click.argument('transformation', type=click.File(mode='r'),
                required=True)
def quickrun(transformation):
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
    print("Loading file...")
    dct = json.load(transformation)
    trans = gufe.Transformation.from_dict(dct)
    print("Planning the campaign...")
    dag = trans.create()
    print("Running the campaign...")
    dagresult = execute(dag)
    print("Done! Analyzing the results....")

    prot_result = trans.protocol.gather([dagresult])
    estimate = prot_result.get_estimate()
    uncertainty = prot_result.get_uncertainty()


    print(f"Here is the result:\ndG = {estimate} ± {uncertainty}\n")
    print("Additional information:")
    for result in dagresult.protocol_unit_results:
        print(f"{result.name}:")
        print(result.outputs)


PLUGIN = OFECommandPlugin(
    command=quickrun,
    section="Simulation",
    requires_ofe=(0, 3)
)
