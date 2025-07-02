import click
from openfecli import OFECommandPlugin
import os

@click.command(
    "view-ligand-network",
    short_help="Visualize a ligand network from a .graphml file."
)
@click.argument(
    "ligand-network",
    type=click.Path(exists=True, readable=True, dir_okay=False,
                    file_okay=True),
)
def view_ligand_network(ligand_network: os.PathLike):
    """
    Visualize a ligand network from a .graphml file.

    e.g. ``openfe view-ligand-network network_setup/ligand_network.graphml``
    
    """
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
    command=view_ligand_network,
    section="Network Planning",
    requires_ofe=(0, 7, 0),
)
