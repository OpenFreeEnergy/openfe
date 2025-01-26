# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import pytest

import MDAnalysis as mda
from openff.units import unit

from openfe.protocols.restraint_utils.geometry.utils import (
    _get_mda_selection,
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
