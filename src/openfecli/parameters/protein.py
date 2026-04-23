# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import sys
from typing import Iterable

import click
from plugcli.params import Option

from openfe import ProteinComponent, ProteinMembraneComponent

_PDB_EXT = [".pdb"]
_PDBX_EXT = [".cif", ".pdbx"]


def _contains_any_substring(input: str, substrings: Iterable[str]) -> bool:
    return any([substring in input for substring in substrings])


def _load_protein_from_file(input_file, protein_class: ProteinComponent | ProteinMembraneComponent):
    valid_ext = _PDB_EXT + _PDBX_EXT
    if not _contains_any_substring(input_file, valid_ext):
        raise ValueError(
            f"To load a {protein_class.__name__}, the file extension must contain one of: {valid_ext}."
        )
    try:
        if _contains_any_substring(input_file, _PDB_EXT):
            return protein_class.from_pdb_file(input_file)
        elif _contains_any_substring(input_file, _PDBX_EXT):
            return protein_class.from_pdbx_file(input_file)
    except ValueError:
        click.secho(f"Unable to load a {protein_class.__name__} from {input_file}.", err=True)
        sys.exit(1)


# TODO: these functions are shims to work with plugcli. We should consider migrating to just click.
def _get_protein(user_input, context):
    return _load_protein_from_file(user_input, ProteinComponent)


def _get_protein_membrane(user_input, context):
    return _load_protein_from_file(user_input, ProteinMembraneComponent)


PROTEIN = Option(
    "-p",
    "--protein",
    help=("ProteinComponent. Can be provided as an PDB or as a PDBx/mmCIF file."),
    getter=_get_protein,
)

PROTEIN_MEMBRANE = Option(
    "--protein-membrane",
    help=("ProteinMembraneComponent. Can be provided as an PDB or as a PDBx/mmCIF file."),
    getter=_get_protein_membrane,
)
