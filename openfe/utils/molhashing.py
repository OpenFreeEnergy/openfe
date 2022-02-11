# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from rdkit import Chem
from typing import NamedTuple


class MoleculeHash(NamedTuple):
    smiles: str
    name: str


def hashmol(mol, name=""):
    # MolToSMILES are canonical by default
    return MoleculeHash(Chem.MolToSmiles(mol), name)
