# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from dataclasses import dataclass
from rdkit import Chem
from typing import Dict, Union, TypeVar

RDKitMol = TypeVar('RDKitMol')


@dataclass
class AtomMapping:
    """Simple container with the mapping between two Molecules

    Attributes
    ----------
    mol1, mol2 : rdkit.Mol
      the two Molecules in the mapping
    mol1_to_mol2 : dict
      maps the index of an atom in either molecule **A** or **B** to the other.
      If this atom has no corresponding atom, None is returned.

    """
    mol1: RDKitMol
    mol2: RDKitMol
    mol1_to_mol2: Dict[int, int]

    def __hash__(self):
        # hash of SMILES of each molecule, and a hash of both mappings
        return hash((Chem.MolToSmiles(self.mol1), Chem.MolToSmiles(self.mol2),
                     tuple(self.mol1_to_mol2.items())))

    @classmethod
    def from_perses(cls, perses_mapping):
        raise NotImplementedError()
