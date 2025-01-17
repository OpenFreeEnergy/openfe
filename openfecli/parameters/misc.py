import click
from plugcli.params import Option

N_PROTOCOL_REPEATS = Option(
    "-n",
    "--n-protocol-repeats",
    type=click.INT,
    help="The number of completely independent repeats per protocol unit.  "\
        "For example, setting ``--n-protocol-repeats=1`` can allow for each individual repeat to be submitted in parallel,"\
        "whereas ``--n-protocol-repeats=3`` would run 3 repeats in serial within a single unit.",
)
