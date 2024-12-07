import click
import pathlib
from plugcli.params import MultiStrategyGetter, Option, NOT_PARSED


def get_file_and_extension(user_input, context):
    file = user_input
    ext = file.name.split('.')[-1] if file else None
    return file, ext


def ensure_file_does_not_exist(value):
    # TODO: I believe we can replace this with click.option(file_okay=False)
    if value and value.exists():
        raise click.BadParameter(f"File '{value}' already exists.")

def ensure_parent_dir_exists(value):
    if value and not value.parent.is_dir():
        raise click.BadParameter(f"Cannot write to {value}, parent directory does not exist.")

def validate_outfile(ctx, param, value):
    ensure_file_does_not_exist(value)
    ensure_parent_dir_exists(value)
    return value

OUTPUT_FILE_AND_EXT = Option(
    "-o", "--output",
    help="output file",
    getter=get_file_and_extension,
    type=click.File(mode='wb'),
)
