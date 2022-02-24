# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from dataclasses import dataclass
from itertools import chain
from typing import Dict, NewType

from rdkit import Chem

from openfe.setup import Molecule

AtomMap = NewType("AtomMap", Dict[int, int])


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
    mol1_to_mol2: AtomMap

    def __hash__(self):
        return hash(
            (hash(self.mol1), hash(self.mol2), tuple(self.mol1_to_mol2.items()))
        )

    @classmethod
    def from_perses(cls, perses_mapping):
        raise NotImplementedError()

    def _match_elements(self, idx: int):
        elem_mol1 = self.mol1.GetAtomWithIdx(idx).GetAtomicNum()
        elem_mol2 = self.mol2.GetAtomWithIdx(self.mol1_to_mol2[idx]).GetAtomicNum()
        if elem_mol1 == elem_mol2:
            return True
        else:
            return False

    def _get_unique_bonds_and_atoms(self, mapping: AtomMap):

        uniques = {
            "atoms": [],  # atoms which fully don't exist in mol2
            "elements": [],  # atoms which exist but change elements in mol2
            "bonds": [],  # bonds involving either unique atoms or elements
        }

        for at in self.mol1.GetAtoms():
            idx = at.GetIdx()
            if idx not in self.mol1_to_mol2.keys():
                uniques["atoms"].append(idx)
            elif not self._match_elements(idx):
                uniques["elements"].append(idx)

        for bond in self.mol1.GetBonds():
            bond_at_idxs = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            for at in chain(uniques["atoms"], uniques["elements"]):
                if at in bond_at_idxs:
                    bond_idx = bond.GetIdx()
                    # avoid double inclusion
                    if bond_idx not in uniques["bonds"]:
                        uniques["bonds"].append(bond_idx)

        return uniques

    def _draw_mapping(self):
        mol1_uniques = self._get_unique_bonds_and_atoms(self.mol1_to_mol2)

        # invert map
        mol2_to_mol1_map = {v: k for k, v in self.mol1_to_mol2.items()}

        mol2_uniques = self._get_unique_bonds_and_atoms(mol2_to_mol1_map)

        atoms_list = [
            mol1_uniques["atoms"] + mol1_uniques["elements"],
            mol2_uniques["atoms"] + mol2_uniques["elements"],
        ]

        # highlight core element changes differently from unique atoms
        red = (220, 50, 32, 0.5)
        blue = (0, 90, 181, 1)

        at1_colours = {}
        for at in mol1_uniques["elements"]:
            at1_colours[at] = blue

        at2_colours = {}
        for at in mol2_uniques["elements"]:
            at2_colours[at] = blue

        atom_colours = [at1_colours, at2_colours]

        bonds_list = [mol1_uniques["bonds"], mol2_uniques["bonds"]]

        d2d = Chem.Draw.rdMolDraw2D.MolDraw2DCairo(600, 300, 300, 300)
        d2d.drawOptions().useBWAtomPalette()
        d2d.drawOptions().continousHighlight = False
        d2d.drawOptions().setHighlightColour(red)
        d2d.drawOptions().addAtomIndices = True
        d2d.DrawMolecules(
            [self.mol1, self.mol2],
            highlightAtoms=atoms_list,
            highlightBonds=bonds_list,
            highlightAtomColors=atom_colours,
        )
        d2d.FinishDrawing()
        return d2d.GetDrawingText()

    def visualize(self):
        """Visualize atom mapping in a Jupyter Notebook"""
        from IPython.display import Image

        return Image(self._draw_mapping())
