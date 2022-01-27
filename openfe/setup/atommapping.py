# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from rdkit import Chem
from typing import Dict, Union, TypeVar

RDKitMol = TypeVar('RDKitMol')


class AtomMapping:
    """Simple container with the mapping between two Molecules

    Attributes
    ----------
    mol1, mol2 : rdkit.Mol
      the two Molecules in the mapping
    mol1_to_mol2, mol2_to_mol1 : dict
      maps the index of an atom in either molecule **A** or **B** to the other.
      If this atom has no corresponding atom, None is returned.


    The size of each molecule is given by the length of the x_to_y dictionary
    """
    mol1: RDKitMol
    mol2: RDKitMol
    mol1_to_mol2: Dict[int, Union[int, None]]
    mol2_to_mol1: Dict[int, Union[int, None]]

    def __init__(self, molA: RDKitMol, molB: RDKitMol, mol1_to_mol2, mol2_to_mol1):
        self.mol1 = molA
        self.mol2 = molB
        self.mol1_to_mol2 = mol1_to_mol2
        self.mol2_to_mol1 = mol2_to_mol1

    def __hash__(self):
        # hash of SMILES of each molecule, and a hash of both mappings
        return hash((Chem.MolToSmiles(self.mol1), Chem.MolToSmiles(self.mol2),
                     tuple(self.mol1_to_mol2.items()), tuple(self.mol2_to_mol1.items())))

    @classmethod
    def from_perses(cls, perses_mapping):
        raise NotImplementedError()
