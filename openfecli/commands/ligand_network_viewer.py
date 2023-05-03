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
    from openfe.utils.atommapping_network_plotting import main
    main(ligand_network)


PLUGIN = OFECommandPlugin(
    command=ligand_network_viewer,
    section="Setup",
    requires_ofe=(0, 7, 0),
)
