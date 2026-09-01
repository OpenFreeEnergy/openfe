from pathlib import Path

import click
from gufe import AlchemicalNetwork

from openfecli import OFECommandPlugin
from openfecli.parameters import ALCHEMICAL_NETWORK
from openfecli.utils import print_duration, write


# TODO: add n_repeats
def setup_task_campaign_main(alchemical_network: AlchemicalNetwork, name: str | None = None):
    from openfe.orchestration.exorcist_utils import build_task_db_from_alchemical_network

    db_path = Path(f"{name}.db")
    # TODO: add progress bar
    _, wh = build_task_db_from_alchemical_network(
        alchemical_network=alchemical_network,
        warehouse_dir=Path(name),
        db_path=db_path,
    )
    write(f"Warehouse written to: {wh.root_dir}")
    write(f"TaskDB written to: {db_path}")


@click.command(
    "setup-task-campaign",
    short_help="Build a Warehouse and corresponding TaskDB from an AlchemicalNetwork.",
)
@ALCHEMICAL_NETWORK.parameter(multiple=False, required=True, help=ALCHEMICAL_NETWORK.kwargs["help"])
@print_duration
def setup_task_campaign(alchemical_network: str | Path):
    # TODO: allow user-supplied name and out_dir
    name = Path(alchemical_network).stem
    write("Loading AlchemicalNetwork ...")
    loaded_alch_net = ALCHEMICAL_NETWORK.get(alchemical_network)
    write("Creating Warehouse and TaskDB ...")
    setup_task_campaign_main(alchemical_network=loaded_alch_net, name=name)


PLUGIN = OFECommandPlugin(
    command=setup_task_campaign, section="Planning & Setup", requires_ofe=(1, 12)
)
