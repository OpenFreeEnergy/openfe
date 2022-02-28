# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from __future__ import annotations

from dataclasses import dataclass
from rdkit import Chem
from typing import Dict

from openfe.setup import Molecule


def _iter_bonded_H(atom):
    # return iterable of hydrogens bonded to a given rdkit Atom
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetAtomicNum() == 1:
            yield other
    return


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

    def to_explicit(self) -> AtomMapping:
        """Add explicit hydrogens to this AtomMapping"""
        rdkit_mol1 = self.mol1.to_rdkit()
        rdkit_mol2 = self.mol2.to_rdkit()

        # rdkit uses 0 as a null value for atommap, so we need to use 1-based
        # indexing here
        for i, atom in enumerate(rdkit_mol1.GetAtoms()):
            atom.SetAtomMapNum(i+1)
        for i, atom in enumerate(rdkit_mol2.GetAtoms()):
            atom.SetAtomMapNum(i+1)

        explicit_mol1 = Chem.AddHs(rdkit_mol1)
        explicit_mol2 = Chem.AddHs(rdkit_mol2)

        newmapping = self.mol1_to_mol2.copy()
        # loop over atoms in the implicit-H mapping,
        # then try and match up hydrogens as best as possible
        for i, j in self.mol1_to_mol2.items():
            atom_i = explicit_mol1.GetAtomWithIdx(i)
            atom_j = explicit_mol2.GetAtomWithIdx(j)

            # TODO: Could probably do better by geometrically matching
            for i_H, j_H in zip(_iter_bonded_H(atom_i),
                                _iter_bonded_H(atom_j)):
                newmapping[i_H.GetIdx()] = j_H.GetIdx()

        return AtomMapping(Molecule(explicit_mol1, name=self.mol1.name),
                           Molecule(explicit_mol2, name=self.mol2.name),
                           mol1_to_mol2=newmapping)

    def __hash__(self):
        return hash((hash(self.mol1), hash(self.mol2),
                     tuple(self.mol1_to_mol2.items())))

    @classmethod
    def from_perses(cls, perses_mapping):
        raise NotImplementedError()
