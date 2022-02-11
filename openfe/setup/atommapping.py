# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from dataclasses import dataclass
from typing import Dict, TypeVar

from openfe.setup import Molecule

RDKitMol = TypeVar('RDKitMol')


@dataclass
class AtomMapping:
    """Simple container with the mapping between two Molecules

    Attributes
    ----------
    mol1, mol2 : Molecule
      the two Molecules in the mapping
    mol1_to_mol2 : dict
      maps the index of an atom in either molecule **A** or **B** to the other.
      If this atom has no corresponding atom, None is returned.

    """
    mol1: Molecule
    mol2: Molecule
    mol1_to_mol2: Dict[int, int]

    def __hash__(self):
        return hash((hash(self.mol1), hash(self.mol2),
                     tuple(self.mol1_to_mol2.items())))

    @classmethod
    def from_perses(cls, perses_mapping):
        raise NotImplementedError()
