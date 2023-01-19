# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from typing import Dict, Set, Tuple, Any, Collection, Optional
from itertools import chain

from rdkit import Chem
from rdkit.Chem import AllChem

from openfe.utils.custom_typing import RDKitMol


# highlight core element changes differently from unique atoms
# RGBA color value needs to be between 0 and 1, so divide by 255
RED = (220 / 255, 50 / 255, 32 / 255, 1.0)
BLUE = (0.0, 90 / 255, 181 / 255, 1.0)


def _match_elements(
        mol1: RDKitMol, idx1: int, mol2: RDKitMol, idx2: int
) -> bool:
    """
    Convenience method to check if elements between two molecules (molA
    and molB) are the same.

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


def _get_unique_bonds_and_atoms(
        mapping: Dict[int, int], mol1: RDKitMol, mol2: RDKitMol
) -> Dict:
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
        ("elements"), deleted bonds ("bond_deletions) and altered bonds
        ("bond_changes) for molecule 1.
    """

    uniques: Dict[str, set] = {
        "atoms": set(),  # atoms which fully don't exist in molB
        "elements": set(),  # atoms which exist but change elements in molB
        "bond_deletions": set(),  # bonds which are removed
        "bond_changes": set(),  # bonds which change
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

                if any(i in uniques["atoms"] for i in bond_at_idxs):
                    uniques["bond_deletions"].add(bond_idx)
                else:
                    uniques["bond_changes"].add(bond_idx)

    return uniques


def _draw_molecules(
    d2d,
    mols: Collection[RDKitMol],
    atoms_list: Collection[Set[int]],
    bonds_list: Collection[Set[int]],
    atom_colors: Collection[Dict[Any, Tuple[float, float, float, float]]],
    bond_colors: Collection[dict[int, tuple[float, float, float, float]]],
    highlight_color: Tuple[float, float, float, float],
    atom_mapping: Optional[Dict[Tuple[int, int], Dict[int, int]]] = None,
) -> str:
    """
    Internal method to visualize a molecule, possibly with mapping info

    Parameters
    ----------
    d2d :
        renderer to draw the molecule; currently we only support
        rdkit.rdMolDraw2D
    mols : Collection[RDKitMol]
        molecules to visualize
    atoms_list: Collection[Set[int]]
        iterable containing one set per molecule in ``mols``, with each set
        containing the indices of the atoms to highlight
    bonds_list: Collection[Set[int]]
        iterable containing one set per molecule in ``mols``, with each set
        containing the indices of the atoms involved in bonds to highlight
    atom_colors: Collection[Dict[Any, Tuple[float, float, float, float]]]
        iterable containing one dict per molecule in ``mols``, with each
        dict containing a mapping of RDKit atom to color, expressed as an
        RGBA tuple, for atoms that need special coloring (e.g., element
        changes)
    bond_colors: Collection[dict[int, tuple[float, float, float, float]]]
        one dict for each molecule, each dict mapping
    highlight_color: Tuple[float, float, float, float]
        RGBA tuple for the default highlight color used in the mapping
        visualization
    atom_mapping: Dict[Tuple[int,int], Dict[int, int]], optional
        used to align the molecules to each othter for clearer visualization.
        The structure contains the indices of the molecules in mols as key
        Tuple[molA, molB] and maps the atom indices via the value Dict[
        molA_atom_idx, molB_atom_idx]
        default None
    """
    # input standardization:
    if atom_mapping is None:
        atom_mapping = {}

    if d2d is None:
        # select default layout based on number of molecules
        grid_x, grid_y = {1: (1, 1), 2: (2, 1), }[len(mols)]
        d2d = Chem.Draw.rdMolDraw2D.MolDraw2DCairo(
            grid_x * 300, grid_y * 300, 300, 300)

    # squash to 2D
    copies = [Chem.Mol(mol) for mol in mols]
    for mol in copies:
        AllChem.Compute2DCoords(mol)

    # mol alignments if atom_mapping present
    for (i, j), atomMap in atom_mapping.items():
        AllChem.AlignMol(
            copies[j], copies[i],
            atomMap=[(k, v) for v, k in atomMap.items()]
        )

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
        highlightBondColors=bond_colors,
    )
    d2d.FinishDrawing()
    return d2d.GetDrawingText()


def draw_mapping(
        mol1_to_mol2: Dict[int, int], mol1: RDKitMol, mol2: RDKitMol, d2d=None
):
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
    bonds_list = [
        mol1_uniques["bond_deletions"] | mol1_uniques["bond_changes"],
        mol2_uniques["bond_deletions"] | mol2_uniques["bond_changes"],
    ]

    at1_colors = {at: BLUE for at in mol1_uniques["elements"]}
    at2_colors = {at: BLUE for at in mol2_uniques["elements"]}
    bd1_colors = {bd: BLUE for bd in mol1_uniques["bond_changes"]}
    bd2_colors = {bd: BLUE for bd in mol2_uniques["bond_changes"]}

    atom_colors = [at1_colors, at2_colors]
    bond_colors = [bd1_colors, bd2_colors]


    return _draw_molecules(
        d2d,
        [mol1, mol2],
        atoms_list=atoms_list,
        bonds_list=bonds_list,
        atom_colors=atom_colors,
        bond_colors=bond_colors,
        highlight_color=RED,
        atom_mapping={(0, 1): mol1_to_mol2},
    )


def draw_one_molecule_mapping(mol1_to_mol2, mol1, mol2, d2d=None):
    """Draw the mapping visualization for a single molecular of a mapping

    This will always draw ``molA``. To draw ``molB``, switch order/invert
    ``molA_to_molB`` mapping.

    See :func:`.draw_mapping` for details on the meaning of different
    colors.

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
    uniques = _get_unique_bonds_and_atoms(mol1_to_mol2, mol1, mol2)
    atoms_list = [uniques["atoms"] | uniques["elements"]]
    bonds_list = [uniques["bond_deletions"] | uniques["bond_changes"]]

    atom_colors = [{at: BLUE for at in uniques["elements"]}]
    bond_colors = [{bd: BLUE for bd in uniques["bond_changes"]}]

    return _draw_molecules(d2d,
                           [mol1],
                           atoms_list,
                           bonds_list,
                           atom_colors,
                           bond_colors,
                           RED,
                           )


def draw_unhighlighted_molecule(mol, d2d=None):
    """
    Draw a molecule without any mapping information.

    This uses the same mechanisms as the mapping visualizations, and so can
    be useful in cases where you want the same visual style both for a plain
    molecule and a molecule with the mapping highlighted.

    Parameters
    ----------
    mol : RDKit.Mol
        RDKit representation of the molecule
    d2d : :class:`rdkit.Chem.Draw.rdMolDraw2D.MolDraw2D`
        Optional MolDraw2D backend to use for visualisation.
    """
    red = (220 / 255, 50 / 255, 32 / 255, 1.0)
    return _draw_molecules(
        d2d,
        [mol],
        atoms_list=[[]],
        bonds_list=[[]],
        atom_colors=[{}],
        bond_colors=[{}],
        highlight_color=red,
    )
