import click
import pathlib

from openfecli import OFECommandPlugin

@click.command('peek', short_help="Print the reduced contents (omitting large data) of a results JSON.")
@click.argument('json', type=click.Path(dir_okay=False, file_okay=True, path_type=pathlib.Path), required=True)
def peek(json:pathlib.Path):
    click.echo(json)

PLUGIN = OFECommandPlugin(command=peek,
                          section="Miscellaneous",
                          requires_ofe=(1,0)
                          )
