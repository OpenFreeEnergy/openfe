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


def _load_molecule_from_sdf(user_input, context):
    from openfe.setup import Molecule
    try:
        with open(user_input, mode="r") as sdf:
            contents = sdf.read()
        return Molecule.from_sdf_string(contents)
    except:  # any exception should try other strategies
        return NOT_PARSED


get_molecule = MultiStrategyGetter(
    strategies=[
        _load_molecule_from_sdf,
        # NOTE: I think loading from smiles must be last choice, because
        # failure will give meaningless user-facing errors
        _load_molecule_from_smiles,
    ],
    error_message="Unable to generate a molecule from '{user_input}'."
)

MOL = Option(
    "--mol",
    help="Molecule. Can be provided as an SDF file or as a SMILES string.",
    getter=get_molecule
)
