import os
import pathlib

import click
from plugcli.params import NOT_PARSED, MultiStrategyGetter, Option


def get_dir(user_input, context):
    dir_path = pathlib.Path(user_input)
    return dir_path


OUTPUT_DIR = Option(
    "-o",
    "--output-dir",
    help="Path to the output directory. ",
    getter=get_dir,
    type=click.Path(file_okay=False, resolve_path=True),
)
