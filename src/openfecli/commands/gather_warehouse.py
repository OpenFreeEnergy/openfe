import os
import pathlib
import sys
from typing import List, Literal

import click
import numpy as np
import pandas as pd

from openfecli import OFECommandPlugin

# from openfecli.utils import rich_print_to_stdout


@click.command(
    "gather-warehouse",
    short_help="Gather results from a Warehouse and return the raw values",
)
@click.argument(
    "warehouse_path",
    nargs=1,  # accept any number of results
    type=click.Path(dir_okay=True, file_okay=False, path_type=pathlib.Path),
    required=True,
)
def gather_warehouse(
    warehouse_path: os.PathLike | str,
):
    """
    .. warning::

        Gathering of results with ``openfe gather-warehouse`` is an experimental feature
        and is subject to change in a future release of openfe!

    Gather simulation results from the `results` store of a Warehouse and output the raw results.

    """
    from openfe.storage.warehouse import FileSystemWarehouse

    msg = "WARNING! Warehouse and Workers are experimental features and subject to change in a future release of openfe."
    click.secho(msg, err=True, fg="yellow")  # fmt: skip

    warehouse = FileSystemWarehouse.from_dir(warehouse_path)
    raw_results = warehouse.gather_all_results()

    # rich_print_to_stdout(df)


PLUGIN = OFECommandPlugin(
    command=gather_warehouse,
    section="Quickrun Executor",
    requires_ofe=(0, 6),
)
