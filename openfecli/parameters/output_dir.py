import os
import click
import pathlib
from plugcli.params import MultiStrategyGetter, Option, NOT_PARSED


def get_dir(user_input, context):
    dir_path = pathlib.Path(user_input)
    return dir_path


OUTPUT_DIR = Option(
    "-o",
    "--output-dir",
    help="output_dir",
    getter=get_dir,
    type=click.Path(file_okay=False, resolve_path=True),
)
