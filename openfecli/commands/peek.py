import click
import os
import pathlib

from openfecli import OFECommandPlugin

# Copied from gather.py - should live in utils but is slow to import right now.
def load_results(fpath:os.PathLike|str)->dict:
    """Load the data from a results JSON into a dict

    Parameters
    ----------
    fpath : os.PathLike | str
        The path to deserialized results.

    Returns
    -------
    dict
        A dict containing data from the results JSON.
    """

    import json
    from gufe.tokenization import JSON_HANDLER
    return json.load(open(fpath, 'r'), cls=JSON_HANDLER.decoder)

@click.command('peek', short_help="Print the reduced contents (omitting large data) of a results JSON.")
@click.argument('json', type=click.Path(dir_okay=False, file_okay=True, path_type=pathlib.Path), required=True)
def peek(json:pathlib.Path):

    data = load_results(json)
    click.echo(data.keys())

PLUGIN = OFECommandPlugin(command=peek,
                          section="Miscellaneous",
                          requires_ofe=(1,0)
                          )
