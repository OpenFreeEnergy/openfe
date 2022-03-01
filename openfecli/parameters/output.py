import click
from plugcli.params import MultiStrategyGetter, Option, NOT_PARSED


def get_file_and_extension(user_input, context):
    file = user_input
    ext = file.name.split('.')[-1] if file else None
    return file, ext


OUTPUT_FILE_AND_EXT = Option(
    "-o", "--output",
    help="output file",
    getter=get_file_and_extension,
    type=click.File(mode='wb'),
)
