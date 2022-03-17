# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from typing import Dict
from itertools import chain
import copy

from rdkit import Chem
from rdkit.Chem import AllChem

from openfe.utils.typing import RDKitMol


def _match_elements(mol1: RDKitMol, idx1: int,
                    mol2: RDKitMol, idx2: int) -> bool:
    """
    Convenience method to check if elements between two molecules (mol1
    and mol2) are the same.

    Parameters
    ----------
    mol1 : RDKit.Mol
        RDKit representation of molecule 1.
    idx1 : int
        Index of atom to check in molecule 1.
    mol2 : RDKit.Mol
        RDKit representation of molecule 2.
    idx2 : int
        Index of atom to check in molecule 2.

    Returns
    -------
    bool
        True if elements are the same, False otherwise.
    """
    elem_mol1 = mol1.GetAtomWithIdx(idx1).GetAtomicNum()
    elem_mol2 = mol2.GetAtomWithIdx(idx2).GetAtomicNum()
    return elem_mol1 == elem_mol2


def _get_unique_bonds_and_atoms(mapping: Dict[int, int],
                                mol1: RDKitMol, mol2: RDKitMol) -> Dict:
    """
    Given an input mapping, returns new atoms, element changes, and
    involved bonds.

    Parameters
    ----------
    mapping : dict of int:int
        Dictionary describing the atom mapping between molecules 1 and 2.
    mol1 : RDKit.Mol
        RDKit representation of molecule 1.
    mol2 : RDKit.Mol
        RDKit representation of molecule 2.

    Returns
    -------
    uniques : dict
        Dictionary containing; unique atoms ("atoms"), new elements
        ("elements"), and bonds involved with either of the previous two
        ("bonds") for molecule 1.
    """

    uniques: Dict[str, set] = {
        "atoms": set(),  # atoms which fully don't exist in mol2
        "elements": set(),  # atoms which exist but change elements in mol2
        "bonds": set(),  # bonds involving either unique atoms or elements
    }

    for at in mol1.GetAtoms():
        idx = at.GetIdx()
        if idx not in mapping:
            uniques["atoms"].add(idx)
        elif not _match_elements(mol1, idx, mol2, mapping[idx]):
            uniques["elements"].add(idx)

    for bond in mol1.GetBonds():
        bond_at_idxs = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        for at in chain(uniques["atoms"], uniques["elements"]):
            if at in bond_at_idxs:
                bond_idx = bond.GetIdx()
                uniques["bonds"].add(bond_idx)

    return uniques


def _draw_molecules(d2d, mols, atoms_list, bonds_list, atom_colors,
                    highlight_color):
    # squash to 2D
    copies = [copy.deepcopy(mol) for mol in mols]
    for mol in copies:
        AllChem.Compute2DCoords(mol)

    # standard settings for our visualization
    d2d.drawOptions().useBWAtomPalette()
    d2d.drawOptions().continousHighlight = False
    d2d.drawOptions().setHighlightColour(highlight_color)
    d2d.drawOptions().addAtomIndices = True
    d2d.DrawMolecules(
        copies,
        highlightAtoms=atoms_list,
        highlightBonds=bonds_list,
        highlightAtomColors=atom_colors,
    )
    d2d.FinishDrawing()
    return d2d.GetDrawingText()


def draw_mapping(mol1_to_mol2: Dict[int, int],
                 mol1: RDKitMol, mol2: RDKitMol, d2d=None):
    """
    Method to visualise the atom map correspondence between two rdkit
    molecules given an input mapping.

    Legend:
        * Red highlighted atoms: unique atoms, i.e. atoms which are not
          mapped.
        * Blue highlighted atoms: element changes, i.e. atoms which are
          mapped but change elements.
        * Red highlighted bonds: any bond which involves at least one
          unique atom or one element change.

    Parameters
    ----------
    mol1_to_mol2 : dict of int:int
        Atom mapping between input molecules.
    mol1 : RDKit.Mol
        RDKit representation of molecule 1
    mol2 : RDKit.Mol
        RDKit representation of molecule 2
    d2d : :class:`rdkit.Chem.Draw.rdMolDraw2D.MolDraw2D`
        Optional MolDraw2D backend to use for visualisation.
    """
    mol1_uniques = _get_unique_bonds_and_atoms(mol1_to_mol2, mol1, mol2)

    # invert map
    mol2_to_mol1_map = {v: k for k, v in mol1_to_mol2.items()}
    mol2_uniques = _get_unique_bonds_and_atoms(mol2_to_mol1_map, mol2, mol1)

    atoms_list = [
        mol1_uniques["atoms"] | mol1_uniques["elements"],
        mol2_uniques["atoms"] | mol2_uniques["elements"],
    ]

    # highlight core element changes differently from unique atoms
    # RGBA color value needs to be between 0 and 1, so divide by 255
    red = (220/255, 50/255, 32/255, 1)
    blue = (0, 90/255, 181/255, 1)

    at1_colors = {at: blue for at in mol1_uniques["elements"]}
    at2_colors = {at: blue for at in mol2_uniques["elements"]}

    atom_colors = [at1_colors, at2_colors]

    bonds_list = [mol1_uniques["bonds"], mol2_uniques["bonds"]]

    # If d2d is None, use MolDraw2DCairo
    if not d2d:
        d2d = Chem.Draw.rdMolDraw2D.MolDraw2DCairo(600, 300, 300, 300)

    return _draw_molecules(
        d2d,
        [mol1, mol2],
        atoms_list=atoms_list,
        bonds_list=bonds_list,
        atom_colors=atom_colors,
        highlight_color=red,
    )


def draw_one_molecule_mapping(mol1_to_mol2, mol1, mol2, d2d=None):
    """Draw the mapping visualization for a single molecular of a mapping

    This will always draw ``mol1``. To draw ``mol2``, switch order/invert
    ``mol1_to_mol2`` mapping.
    """
    uniques = _get_unique_bonds_and_atoms(mol1_to_mol2, mol1, mol2)
    atoms_list = [uniques["atoms"] | uniques["elements"]]
    bonds_list = [uniques["bonds"]]
    red = (220/255, 50/255, 32/255, 1)
    blue = (0, 90/255, 181/255, 1)

    atom_colors = [{at: blue for at in uniques["elements"]}]

    if d2d is None:
        # TODO: we should try to be smarter about the default size here
        d2d = Chem.Draw.rdMolDraw2D.MolDraw2DCairo(300, 300, 300, 300)

    return _draw_molecules(d2d, [mol1], atoms_list, bonds_list, atom_colors,
                           red)


def draw_unhighlighted_molecule(mol, d2d=None):
    if d2d is None:
        # TODO: we should try to be smarter about the default size here
        d2d = Chem.Draw.rdMolDraw2D.MolDraw2DCairo(300, 300, 300, 300)

    red = (220/255, 50/255, 32/255, 1)
    return _draw_molecules(
        d2d,
        [mol],
        atoms_list=[[]],
        bonds_list=[[]],
        atom_colors=[{}],
        highlight_color=red,
    )
