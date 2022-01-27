# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from rdkit import Chem
from typing import Dict, Union, TypeVar

RDKitMol = TypeVar('RDKitMol')


class AtomMapping:
    """Simple container with the mapping between two Molecules

    Attributes
    ----------
    molA, molB : rdkit.Mol
      the two Molecules in the mapping
    AtoB, BtoA : dict
      maps the index of an atom in either molecule **A** or **B** to the other.
      If this atom has no corresponding atom, None is returned.


    The size of molecule A/B is given by the length of the AtoB/BtoA dictionary
    """
    molA: RDKitMol
    molB: RDKitMol
    AtoB: Dict[int, Union[int, None]]
    BtoA: Dict[int, Union[int, None]]

    def __init__(self, molA: RDKitMol, molB: RDKitMol):
        self.molA = molA
        self.molB = molB
        self.AtoB = dict()
        self.BtoA = dict()

    def __hash__(self):
        # hash of SMILES of each molecule, and a hash of both mappings
        return hash((Chem.MolToSmiles(self.molA), Chem.MolToSmiles(self.molB),
                     tuple(self.AtoB.items()), tuple(self.BtoA.items())))

    @classmethod
    def from_perses(cls, perses_mapping):
        raise NotImplementedError()
