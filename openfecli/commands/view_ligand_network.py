import click
import konnektor
from openfecli import OFECommandPlugin

@click.command(
    "view-ligand-network",
    short_help="Visualize a ligand network"
)
@click.argument(
    "ligand-network",
    type=click.Path(exists=True, readable=True, dir_okay=False,
                    file_okay=True),
)
def view_ligand_network(ligand_network):
    from konnektor.visualization import draw_ligand_network, draw_network_widget

    from openfe.setup import LigandNetwork
    import matplotlib

    matplotlib.use("TkAgg")
    with open(ligand_network) as f:
        graphml = f.read()

    network = LigandNetwork.from_graphml(graphml)
    fig = draw_ligand_network(network, node_size=5000,)
    matplotlib.pyplot.show()


PLUGIN = OFECommandPlugin(
    command=view_ligand_network,
    section="Network Planning",
    requires_ofe=(0, 7, 0),
)
