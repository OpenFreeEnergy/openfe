import os

import click

from openfecli import OFECommandPlugin


@click.command("view-ligand-network", short_help="Visualize a ligand network from a .graphml file.")
@click.argument(
    "ligand-network",
    type=click.Path(exists=True, readable=True, dir_okay=False, file_okay=True),
)
def view_ligand_network(ligand_network: os.PathLike):
    """
    Visualize a ligand network from a .graphml file.

    e.g. ``openfe view-ligand-network network_setup/ligand_network.graphml``

    """
    import matplotlib

    from openfe.setup import LigandNetwork
    from openfe.utils.atommapping_network_plotting import plot_atommapping_network

    matplotlib.use("TkAgg")
    with open(ligand_network) as f:
        graphml = f.read()

    network = LigandNetwork.from_graphml(graphml)
    fig = plot_atommapping_network(network)
    axes = fig.axes
    for ax in axes:
        ax.set_frame_on(False)  # remove the black frame
        for t in ax.texts:
            t.set_clip_on(False)  # do not clip the label in the network plot
    matplotlib.pyplot.show()


PLUGIN = OFECommandPlugin(
    command=view_ligand_network,
    section="Network Planning",
    requires_ofe=(0, 7, 0),
)
