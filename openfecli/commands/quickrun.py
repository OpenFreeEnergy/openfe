# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import click
from openfecli import OFECommandPlugin
import json


@click.command(
    'quickrun',
    short_help="Run a given transformation, saved as a JSON file"
)
@click.argument('transformation', type=click.File(mode='r'))
def quickrun(transformation):
    """Run (in serial) the given transformation.
    """
    import gufe
    from gufe.protocols.protocoldag import execute
    dct = json.load(transformation)
    trans = gufe.Transformation.from_dict(dct)
    dag = trans.create()
    dagresult = execute(dag)

    prot_result = trans.protocol.gather([dagresult])
    estimate = prot_result.get_estimate()
    uncertainty = prot_result.get_uncertainty()

    print(f"dG = {estimate} ± {uncertainty}\n")
    print("Additional information:")
    for result in result.protocol_unit_results:
        print(f"{result.name}:")
        print(result.outputs)


PLUGIN = OFECommandPlugin(
    command=quickrun,
    section="Simulation",
    requires_ofe=(0, 3)
)
