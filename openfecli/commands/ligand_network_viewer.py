import click
from openfecli import OFECommandPlugin

@click.command(
    "ligand-network-viewer",
    short_help="Visualize a ligand network"
)
@click.argument(
    "ligand-network",
    type=click.Path(exists=True, readable=True, dir_okay=False,
                    file_okay=True),
)
def ligand_network_viewer(ligand_network):
    from openfe.utils.atommapping_network_plotting import (
        plot_atommapping_network
    )
    from openfe.setup import LigandNetwork
    import matplotlib

    matplotlib.use("TkAgg")
    with open(ligand_network) as f:
        graphml = f.read()

    network = LigandNetwork.from_graphml(graphml)
    fig = plot_atommapping_network(network)
    matplotlib.pyplot.show()


PLUGIN = OFECommandPlugin(
    command=ligand_network_viewer,
    section="Network Planning",
    requires_ofe=(0, 7, 0),
)
