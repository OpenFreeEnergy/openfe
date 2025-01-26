# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pytest

import itertools
from rdkit import Chem
import MDAnalysis as mda
from openff.units import unit
import numpy as np

from openfe.protocols.restraint_utils.geometry.utils import (
    _get_mda_selection,
    get_aromatic_rings,
    get_aromatic_atom_idxs,
    get_heavy_atom_idxs,
    _wrap_angle,
    check_dihedral_bounds,
)



@pytest.fixture(scope='module')
def eg5_pdb_universe(eg5_protein_pdb):
    return mda.Universe(eg5_protein_pdb)


def test_mda_selection_none_error(eg5_pdb_universe):
    with pytest.raises(ValueError, match="one of either"):
        _ = _get_mda_selection(eg5_pdb_universe)


def test_mda_selection_both_args_error(eg5_pdb_universe):
    with pytest.raises(ValueError, match="both atom_list and"):
        _ = _get_mda_selection(
            eg5_pdb_universe,
            atom_list=[0, 1, 2, 3],
            selection="all"
        )


def test_mda_selection_universe_atom_list(eg5_pdb_universe):
    test_ag = _get_mda_selection(eg5_pdb_universe, atom_list=[0, 1, 2])
    assert eg5_pdb_universe.atoms[[0, 1, 2]] == test_ag


def test_mda_selection_atomgroup_string(eg5_pdb_universe):
    test_ag = _get_mda_selection(eg5_pdb_universe.atoms, selection='all')
    assert test_ag == eg5_pdb_universe.atoms


@pytest.mark.parametrize('smiles, expected', [
    ['C1CCCCC1', []],
    ['[C@@H]1([C@@H]([C@@H](OC([C@@H]1O)O)C(=O)O)O)O', []],
    ['C1=CC=CC=C1', [6]],
    ['C1=CC2C=CC1C=C2', [8]],
    ['C1CC2=CC=CC=C2C1', [6]],
    ['C1=COC=C1', [5]],
    ['C1=CC=C2C=CC=CC2=C1', [10]],
    ['C1=CC=C(C=C1)C2=CC=CC=C2', [6, 6]],
    ['C1=CC=C(C=C1)C(C2=CC=CC=C2)(C3=CC=CC=C3Cl)N4C=CN=C4', [6, 6, 6, 5]]
])
def test_aromatic_rings(smiles, expected):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

    # get the rings
    rings = get_aromatic_rings(mol)

    # check we have the right number of rings & their size
    for i, r in enumerate(rings):
        assert len(r) == expected[i]

    # check that there is no overlap in atom between each ring
    for x, y in itertools.combinations(rings, 2):
        assert x.isdisjoint(y)

    # get the aromatic idx
    arom_idxs = get_aromatic_atom_idxs(mol)

    # Check that all the ring indices are aromatic
    assert all(idx in arom_idxs for idx in itertools.chain(*rings))

    # Also check the lengths match
    assert sum(len(r) for r in rings) == len(arom_idxs)

    # Finallly check that all the arom_idxs are actually aromatic
    for idx in arom_idxs:
        at = mol.GetAtomWithIdx(idx)
        assert at.GetIsAromatic()

@pytest.mark.parametrize('smiles, nheavy, nlight', [
    ['C1CCCCC1', 6, 12],
    ['[C@@H]1([C@@H]([C@@H](OC([C@@H]1O)O)C(=O)O)O)O', 13, 10],
    ['C1=CC=CC=C1', 6, 6],
    ['C1=CC2C=CC1C=C2', 8, 8],
    ['C1CC2=CC=CC=C2C1', 9, 10],
    ['C1=COC=C1', 5, 4],
    ['C1=CC=C2C=CC=CC2=C1', 10, 8],
    ['C1=CC=C(C=C1)C2=CC=CC=C2', 12, 10],
    ['C1=CC=C(C=C1)C(C2=CC=CC=C2)(C3=CC=CC=C3Cl)N4C=CN=C4', 25, 17]
])
def test_heavy_atoms(smiles, nheavy, nlight):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

    n_atoms = len(list(mol.GetAtoms()))

    heavy_atoms = get_heavy_atom_idxs(mol)

    # check all the heavy atoms are indeed heavy
    for idx in heavy_atoms:
        at = mol.GetAtomWithIdx(idx)
        assert at.GetAtomicNum() > 1

    assert len(heavy_atoms) == nheavy
    assert n_atoms == nheavy + nlight


def test_wrap_angle_degrees():
    for i in range(0, 361, 1):
        angle = _wrap_angle(i * unit.degrees)
        if i > 180:
            expected = ((i - 360) * unit.degrees).to('radians').m
        else:
            expected = (i * unit.degrees).to('radians').m

        assert angle.m == pytest.approx(expected)


@pytest.mark.parametrize('angle, expected', [
    [0 * unit.radians, 0 * unit.radians],
    [1 * unit.radians, 1 * unit.radians],
    [4 * unit.radians, 4 - (2 * np.pi) * unit.radians],
    [-4 * unit.radians, -4 + (2 * np.pi) * unit.radians],
])
def test_wrap_angle_radians(angle, expected):
    assert _wrap_angle(angle) == pytest.approx(expected)


@pytest.mark.parametrize('dihed, expected', [
    [3 * unit.radians, False],
    [0 * unit.radians, True],
    [-3 * unit.radians, False],
    [300 * unit.degrees, True],
    [181 * unit.degrees, False],
])
def test_check_dihedral_bounds(dihed, expected):
    ret = check_dihedral_bounds(dihed)
    assert ret == expected


@pytest.mark.parametrize('dihed, lower, upper, expected', [
    [3 * unit.radians, -3.1 * unit.radians, 3.1 * unit.radians, True],
    [300 * unit.degrees, -61 * unit.degrees, 301 * unit.degrees, True],
    [300 * unit.degrees, 299 * unit.degrees, -61 * unit.degrees, False]
])
def test_check_dihedral_bounds_defined(dihed, lower, upper, expected):
    ret = check_dihedral_bounds(
        dihed, lower_cutoff=lower, upper_cutoff=upper
    )
    assert ret == expected
