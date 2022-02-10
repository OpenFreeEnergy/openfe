# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from collections import namedtuple
from rdkit import Chem


class MoleculeHash(namedtuple):
    smiles: str
    name: str


def hashmol(mol):
    # canonical by default
    try:
        name = mol.GetProp("_Name")
    except KeyError:
        name = ""
    return MoleculeHash(Chem.MolToSmiles(mol), name)
