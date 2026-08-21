from pathlib import Path

import click
from gufe import AlchemicalNetwork

from openfecli import OFECommandPlugin
from openfecli.parameters import ALCHEMICAL_NETWORK
from openfecli.utils import print_duration, write


def build_warehouse_main(alchemical_network: AlchemicalNetwork, name: str | None = None):
    from openfe.orchestration.exorcist_utils import build_task_db_from_alchemical_network

    db_path = Path(f"{name}.db")
    # TODO: add progress bar
    _, wh = build_task_db_from_alchemical_network(
        alchemical_network=alchemical_network,
        warehouse_dir=Path(name),
        db_path=db_path,
    )
    write(f"created Warehouse at {wh.root_dir}")
    write(f"created TaskDB at {db_path}")


@click.command(
    "build-warehouse",
    short_help="Build a Warehouse and corresponding TaskDB from an AlchemicalNetwork.",
)
@ALCHEMICAL_NETWORK.parameter(multiple=False, required=True, help=ALCHEMICAL_NETWORK.kwargs["help"])
@print_duration
def build_warehouse(alchemical_network: str | Path):
    name = Path(alchemical_network).stem
    loaded_alch_net = ALCHEMICAL_NETWORK.get(alchemical_network)
    build_warehouse_main(alchemical_network=loaded_alch_net, name=name)


PLUGIN = OFECommandPlugin(command=build_warehouse, section="Network Planning", requires_ofe=(1, 12))
