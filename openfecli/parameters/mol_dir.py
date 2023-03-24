# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import glob
from rdkit import Chem
from plugcli.params import MultiStrategyGetter, Option, NOT_PARSED

def _load_molecules_from_sdf(user_input, context):
    sdfs = glob.glob(user_input+"/*.sdf")
    if len(sdfs) == 0:  # this silences some stderr spam
        return NOT_PARSED

    from openfe import SmallMoleculeComponent
    mols = []
    for sdf in sdfs:
        try:
            mols.append(SmallMoleculeComponent.from_sdf_file(sdf))
        except ValueError:  # any exception should try other strategies
            return NOT_PARSED
    return mols

def _load_molecules_from_mol2(user_input, context):
    mol2s = glob.glob(user_input+"/*.mol2")
    if len(mol2s) == 0:  # this silences some stderr spam
        return NOT_PARSED

    from openfe import SmallMoleculeComponent
    mols = []
    for mol2 in mol2s:
        try:
            rdmol = Chem.MolFromMol2File(mol2, removeHs=False)
            mols.append(SmallMoleculeComponent(rdkit=rdmol))
        except ValueError:  # any exception should try other strategies
            return NOT_PARSED
    return mols


get_molecules = MultiStrategyGetter(
    strategies=[
        _load_molecules_from_sdf,
        _load_molecules_from_mol2
    ],
    error_message="Unable to generate a molecule from '{user_input}'."
)

MOL_DIR = Option(
    "-m", "--mol-dir",
    help=("SmallMoleculeComponents from a folder. Folder needs to contain SDF/MOL2 files"
          " string."),
    getter=get_molecules
)
