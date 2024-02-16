import pathlib

import click
from plugcli.params import NOT_PARSED, MultiStrategyGetter, Option


def get_file_and_extension(user_input, context):
    file = user_input
    ext = file.name.split(".")[-1] if file else None
    return file, ext


def ensure_file_does_not_exist(ctx, param, value):
    if value and value.exists():
        raise click.BadParameter(f"File '{value}' already exists.")
    return value


OUTPUT_FILE_AND_EXT = Option(
    "-o",
    "--output",
    help="output file",
    getter=get_file_and_extension,
    type=click.File(mode="wb"),
)
