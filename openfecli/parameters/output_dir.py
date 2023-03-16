import click
import pathlib
from plugcli.params import MultiStrategyGetter, Option, NOT_PARSED


def get_dir(user_input, context):
    dir_path = user_input
    return dir_path


def ensure_file_does_not_exist(ctx, param, value):
    if value and value.exists():
        raise click.BadParameter(f"File '{value}' already exists.")
    return value


OUTPUT_DIR = Option(
    "-od", "--output_dir",
    help="output_dir",
    getter=get_dir,
    type=click.Path(file_okay=False, resolve_path=True),
)
