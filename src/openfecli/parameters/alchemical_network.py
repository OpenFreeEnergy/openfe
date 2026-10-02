import click
import gufe
from plugcli.params import Option


def alchemical_network_getter(user_input, context):
    return gufe.AlchemicalNetwork.from_json(user_input)


ALCHEMICAL_NETWORK = Option(
    "--alchemical-network",
    type=click.Path(exists=True),
    help=("Path to a JSON file containing a serialized AlchemicalNetwork."),
    getter=alchemical_network_getter,
)
