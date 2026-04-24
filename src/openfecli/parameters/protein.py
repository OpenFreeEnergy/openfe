# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

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
    info_str = (
        f"Unable to load a {protein_class.__name__} from {click.format_filename(input_file)}: "
    )
    if not _contains_any_substring(input_file, valid_ext):
        raise click.BadParameter(info_str + f"File extension must contain one of: {valid_ext}.")
    try:
        if _contains_any_substring(input_file, _PDB_EXT):
            return protein_class.from_pdb_file(input_file)
        elif _contains_any_substring(input_file, _PDBX_EXT):
            return protein_class.from_pdbx_file(input_file)
    except ValueError as e:
        raise click.BadParameter(info_str + f"{e}")


# TODO: these functions are shims to work with plugcli. We should consider migrating to just click.
def _get_protein(user_input, context):
    return _load_protein_from_file(user_input, ProteinComponent)


def _get_protein_membrane(user_input, context):
    return _load_protein_from_file(user_input, ProteinMembraneComponent)


PROTEIN = Option(
    "-p",
    "--protein",
    help=(
        "Path to a PDB or PDBx/mmCIF file containing a protein. Mutually exclusive with --protein-membrane."
    ),
    getter=_get_protein,
)

PROTEIN_MEMBRANE = Option(
    "--protein-membrane",
    help=(
        'Path to a PDB or PDBx/mmCIF file containing a fully solvated protein-membrane system. Mutually exclusive with --protein. See "Combining System Components into a Single PDB File"'
    ),
    getter=_get_protein_membrane,
)
