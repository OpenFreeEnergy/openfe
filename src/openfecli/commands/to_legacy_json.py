from pathlib import Path

import click

from openfecli import OFECommandPlugin
from openfecli.parameters import OUTPUT_DIR  #  OVERWRITE, WAREHOUSE
from openfecli.utils import print_duration, write


def to_legacy_json_main(warehouse_path: Path, output_dir: Path):
    from openfe.storage.utils import warehouse_to_legacy_json
    from openfe.storage.warehouse import FileSystemWarehouse

    write("Loading results from Warehouse ...")
    warehouse = FileSystemWarehouse.from_dir(warehouse_path)
    result_edges = warehouse.gather_all_results()

    if len(result_edges) == 0:
        write(f"No results found in {warehouse_path}.")
    else:
        warehouse_to_legacy_json(result_edges, out_dir=output_dir)
        write(
            f"Results saved to {output_dir}. You may now run ``openfe gather`` on this directory, if applicable."
        )


@click.command(
    "to-legacy-json",
    short_help="Write out the results from a Warehouse directory to JSON files compatible with openfe gather.",
)
@click.argument(
    "warehouse_path",
    type=click.Path(
        exists=True,
        readable=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@OUTPUT_DIR.parameter(help=OUTPUT_DIR.kwargs["help"] + " Defaults to `./results/", default="results/")  # fmt: skip
@print_duration
def to_legacy_json(warehouse_path: str, output_dir: str):
    """
    Execute one available task from a warehouse task graph.

    The warehouse directory must contain a ``tasks.db`` task database and task
    payloads under ``tasks/`` created via OpenFE orchestration setup.
    """
    msg = "WARNING! This is an experimental feature and subject to change in a future release of openfe."
    click.secho(msg, err=True, fg="yellow")  # fmt: skip
    to_legacy_json_main(warehouse_path=warehouse_path, output_dir=output_dir)


PLUGIN = OFECommandPlugin(command=to_legacy_json, section="Results Gathering", requires_ofe=(1, 13))
