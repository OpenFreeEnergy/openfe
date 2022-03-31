# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from plugcli.params import MultiStrategyGetter, Option, NOT_PARSED


def _load_molecule_from_smiles(user_input, context):
    from openfe.setup import SmallMoleculeComponent
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
    return SmallMoleculeComponent(rdkit=mol)


def _load_molecule_from_sdf(user_input, context):
    if '.sdf' not in str(user_input):  # this silences some stderr spam
        return NOT_PARSED

    from openfe.setup import SmallMoleculeComponent
    try:
        return SmallMoleculeComponent.from_sdf_file(user_input)
    except ValueError:  # any exception should try other strategies
        return NOT_PARSED


def _load_molecule_from_mol2(user_input, context):
    if '.mol2' not in str(user_input):
        return NOT_PARSED

    from rdkit import Chem
    from openfe.setup import SmallMoleculeComponent

    m = Chem.MolFromMol2File(user_input)
    if m is None:
        return NOT_PARSED
    else:
        return SmallMoleculeComponent(m)


get_molecule = MultiStrategyGetter(
    strategies=[
        _load_molecule_from_sdf,
        _load_molecule_from_mol2,
        # NOTE: I think loading from smiles must be last choice, because
        # failure will give meaningless user-facing errors
        _load_molecule_from_smiles,
    ],
    error_message="Unable to generate a molecule from '{user_input}'."
)

MOL = Option(
    "--mol",
    help=("SmallMoleculeComponent. Can be provided as an SDF file or as a SMILES "
          " string."),
    getter=get_molecule
)
