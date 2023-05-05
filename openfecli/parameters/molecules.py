# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import click
import glob
import itertools
import pathlib

from plugcli.params import MultiStrategyGetter, Option, NOT_PARSED

# MOVE TO GUFE ####################################################

def molecule_getter(user_input, context):
    from openfe import load_molecules
    return load_molecules(user_input)

MOL_DIR = Option(
    "-M",
    "--molecules",
    type=click.Path(exists=True),
    help=(
        "A directory or file containing all molecules to be loaded, either"
        " as a single SDF or multiple MOL2/SDFs."
    ),
    getter=molecule_getter,
)
