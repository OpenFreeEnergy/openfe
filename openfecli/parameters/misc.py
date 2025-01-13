import click
from plugcli.params import Option

N_PROTOCOL_REPEATS = Option(
    "-n",
    "--n-protocol-repeats",
    type=click.INT,
    help="The number of completely independent repeats of the entire sampling process.",
)
