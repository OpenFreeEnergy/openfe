import click
from openfecli import OFECommandPlugin

@click.command(
    "ligand-network-viewer",
    short_help="Visualize a ligand network"
)
@click.argument(
    "ligand_network",
    type=click.Path(exists=True, readable=True, is_file=True),
    help="graphml file for the ligand network",
)
def ligand_network_viewer(ligand_network):
    from openfe.utils.atommapping_network_plotting import main
    main(ligand_network)
