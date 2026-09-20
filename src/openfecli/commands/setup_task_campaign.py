from pathlib import Path

import click
from gufe import AlchemicalNetwork

from openfecli import OFECommandPlugin
from openfecli.parameters import ALCHEMICAL_NETWORK
from openfecli.utils import print_duration, write


# TODO: add n_repeats
def setup_task_campaign_main(alchemical_network: AlchemicalNetwork, name: str | None = None):
    from openfe.orchestration.exorcist_utils import setup_task_campaign

    db_path = Path(f"tasks_{name}.db")
    # TODO: add progress bar
    _, wh = setup_task_campaign(
        alchemical_network=alchemical_network,
        warehouse_dir=Path(f"warehouse_{name}"),
        db_path=db_path,
    )
    write(f"Warehouse written to: {wh.root_dir}")
    write(f"TaskDB written to: {db_path}")


@click.command(
    "setup-task-campaign",
    short_help="Build a Warehouse and corresponding TaskDB from an AlchemicalNetwork.",
)
# TODO: add --name and --amend options
@ALCHEMICAL_NETWORK.parameter(multiple=False, required=True, help=ALCHEMICAL_NETWORK.kwargs["help"])
@print_duration
def setup_task_campaign(alchemical_network: str | Path):
    """From an AlchemicalNetwork, create the necessary objects for task-based execution:
    - 'warehouse_{name}/': a Warehouse (on the local filesystem as a directory) that stores all setup, task, and results data.
    - 'tasks_{name}.db': a sqlite TaskDB that tracks orchestration status of the tasks.
    - 'scratch/': a local directory used as temporary storage during simulation execution.

    """
    # TODO: allow user-supplied name and out_dir?
    name = Path(alchemical_network).stem
    write("Loading AlchemicalNetwork ...")
    loaded_alch_net = ALCHEMICAL_NETWORK.get(alchemical_network)
    write("Creating Warehouse and TaskDB ...")
    setup_task_campaign_main(alchemical_network=loaded_alch_net, name=name)


PLUGIN = OFECommandPlugin(
    command=setup_task_campaign, section="Planning & Setup", requires_ofe=(1, 12)
)
