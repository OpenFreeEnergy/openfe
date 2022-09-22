# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import click
from openfecli import OFECommandPlugin
import json


@click.command(
    'quickrun',
    short_help="Run an edge from a saved DAG file"
)
@click.argument('dagfile', type=click.File(mode='r'))
@click.option('-o', '--outfile', type=click.File(mode='w'))
def quickrun(dagfile, outfile):
    """Run (in serial) the DAG associated with a given edge.
    """
    from gufe.protocols.protocoldag import execute, ProtocolDAG
    dct = json.load(dagfile)
    dag = ProtocolDAG.from_dict(dct)
    result = execute(dag)

    # currently pathlib.Path can't be serialized
    # outfile.write(json.dumps(result.to_dict()))

    # remove these once we can serialize things
    for result in result.protocol_unit_results:
        print(f"{result.name}:")
        print(result.outputs)


PLUGIN = OFECommandPlugin(
    command=quickrun,
    section="Simulation",
    requires_ofe=(0, 3)
)
