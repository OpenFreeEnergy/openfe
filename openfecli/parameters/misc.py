import click
from plugcli.params import Option

N_PROTOCOL_REPEATS = Option(
    "--n-protocol-repeats",
    type=click.INT,
    help="Number of independent repeats(s) to be run per execution of a transformation using the `openfe quickrun` command.\n\n"
    "For example:\n\n  `--n-protocol-repeats=3` means `openfe quickrun` will execute 3 repeats in serial.\n\n"
    "  `--n-protocol-repeats=1` means `openfe quickrun` will execute only 1 repeat per call, "
    "which allows for individual repeats to be submitted in parallel by calling `openfe quickrun` on the same input JSON file multiple times.",
)

NCORES = Option(
    "-n",
    "--n-cores",
    help="The number of cores which should be used for multiprocessing stages."
)

OVERWRITE = Option(
    "--overwrite-charges",
    is_flag=True,
    default=False,
    help="If the charges already present in the molecules should be overwritten."
)