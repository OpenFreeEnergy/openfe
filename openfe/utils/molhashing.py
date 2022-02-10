# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from rdkit import Chem
from typing import NamedTuple


class MoleculeHash(NamedTuple):
    smiles: str
    name: str


def hashmol(mol):
    # canonical by default
    try:
        name = mol.GetProp("_Name")
    except KeyError:
        name = ""
    return MoleculeHash(Chem.MolToSmiles(mol), name)
