import click
from gufe import AlchemicalNetwork

from openfecli import OFECommandPlugin
from openfecli.parameters import ALCHEMICAL_NETWORK
from openfecli.utils import print_duration, write


def build_warehouse_main(alchemical_network: AlchemicalNetwork):
    from openfe.orchestration.exorcist_utils import build_task_db_from_alchemical_network

    task_db, wh = build_task_db_from_alchemical_network(
        alchemical_network, warehouse_dir="campaign/", db_path="campaign.db"
    )
    write(f"created Warehouse at {wh.root_dir}")
    # write('created TaskDB at {}')


@click.command(
    "build-warehouse",
    short_help="Build a Warehouse and corresponding TaskDB from an AlchemicalNetwork.",
)
@ALCHEMICAL_NETWORK.parameter(help=ALCHEMICAL_NETWORK.kwargs["help"], required=True)
@print_duration
def build_warehouse(alchemical_network: AlchemicalNetwork):
    build_warehouse_main(alchemical_network)


PLUGIN = OFECommandPlugin(command=build_warehouse, section="Network Planning", requires_ofe=(1, 12))
