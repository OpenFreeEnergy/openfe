# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from plugcli.params import MultiStrategyGetter, Option, NOT_PARSED


def _load_protein_from_pdb(user_input, context):
    if '.pdb' not in str(user_input):  # this silences some stderr spam
        return NOT_PARSED

    from openfe.setup import ProteinComponent
    try:
        return ProteinComponent.from_pdb_file(user_input)
    except ValueError:  # any exception should try other strategies
        return NOT_PARSED

def _load_protein_from_pdbx(user_input, context):
    if '.pdbx' not in str(user_input):  # this silences some stderr spam
        return NOT_PARSED

    from openfe.setup import ProteinComponent
    try:
        return ProteinComponent.from_pdbx_file(user_input)
    except ValueError:  # any exception should try other strategies
        return NOT_PARSED
  
    
get_molecule = MultiStrategyGetter(
    strategies=[
        _load_protein_from_pdb,
        _load_protein_from_pdbx,
    ],
    error_message="Unable to generate a molecule from '{user_input}'."
)

PROTEIN = Option(
    "--protein",
    help=("ProteinComponent. Can be provided as an PDB file or as a PDBx/mmCIF file. "
          " string."),
    getter=get_molecule
)
