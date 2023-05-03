import click
import pathlib
from plugcli.params import MultiStrategyGetter, Option, NOT_PARSED


def get_file_and_extension(user_input, context):
    file = user_input
    ext = file.name.split('.')[-1] if file else None
    return file, ext


def ensure_file_does_not_exist(ctx, param, value):
    if value and value.exists():
        raise click.BadParameter(f"File '{value}' already exists.")
    return value


OUTPUT_FILE_AND_EXT = Option(
    "-o", "--output",
    help="output file",
    getter=get_file_and_extension,
    type=click.File(mode='wb'),
)

LIGAND_NETWORK = Option(
    "-L", "--ligand-network",
    help="output ligand network graphml",
    type=click.File(mode="w"),
    getter=lambda x, ctx: x
)
