# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from plugcli.params import MultiStrategyGetter, Option, NOT_PARSED

def _load_molecule_from_smiles(user_input, context):
    from openfe.setup import Molecule
    from rdkit import Chem
    # MolFromSmiles returns None if the string is not a molecule
    # TODO: find some way to redirect the error messages? Messages stayed
    # after either redirect_stdout or redirect_stderr.
    mol = Chem.MolFromSmiles(user_input)
    if mol is None:
        return NOT_PARSED

    # TODO: next is (temporary?) hack: see
    # https://github.com/OpenFreeEnergy/Lomap/issues/4
    Chem.rdDepictor.Compute2DCoords(mol)
    return Molecule(rdkit=mol)


get_molecule = MultiStrategyGetter(
    strategies=[
        # NOTE: I think loading from smiles must be last choice, because
        # failure will give meaningless user-facing errors
        _load_molecule_from_smiles,
    ],
    error_message="Unable to generate a molecule from '{user_input}'."
)

MOL = Option(
    "--mol",
    help="Molecule",
    getter=get_molecule
)
