# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import numpy as np
import pytest

from openfe.protocols.restraint_utils.geometry.boresch.guest import (
    _bonded_angles_from_pool,
    _sort_by_distance_from_atom,
    _get_guest_atom_pool,
    find_guest_atom_candidates,
)
from openfe.protocols.restraint_utils.geometry.utils import (
    get_aromatic_rings,
    get_heavy_atom_idxs,
)
from openff.units import unit
import MDAnalysis as mda

@pytest.mark.parametrize("aromatic, expected", [
    pytest.param(True, [(17, 15, 13), (17, 31, 33)], id="Aromatic"),
    pytest.param(False, [(17, 15, 13), (17, 18, 19), (17, 18, 23), (17, 18, 27), (17, 31, 33)], id="Mixture")
])
def test_get_bond_angle_aromatic(eg5_ligands, aromatic, expected):
    """
    Make sure non-aromatic atoms are correctly filtered when requested
    """
    rd_mol = eg5_ligands[0].to_rdkit()
    angles = _bonded_angles_from_pool(
        rdmol=rd_mol,
        atom_idx=17,
        atom_pool=[a.GetIdx() for a in rd_mol.GetAtoms() if a.GetAtomicNum() > 1],
        aromatic_only=aromatic
    )
    assert angles == expected


def test_sort_by_distance(eg5_ligands):

    rd_mol = eg5_ligands[0].to_rdkit()
    sorted_atoms = _sort_by_distance_from_atom(
        rdmol=rd_mol,
        target_idx=33,
        atom_idxs=[1, 12, 18]
    )
    assert sorted_atoms == [12, 18, 1]


def test_get_guest_atom_pool(eg5_ligands):

    rd_mol = eg5_ligands[0].to_rdkit()
    rmsf = np.zeros(rd_mol.GetNumAtoms()) * unit.nanometer
    pool_atoms, rings = _get_guest_atom_pool(
        rdmol=rd_mol,
        rmsf=rmsf,
        rmsf_cutoff=0.1 * unit.nanometer
    )
    # make sure only rings were found
    assert rings

    # make sure all aromatic ring atoms are found
    rings = get_aromatic_rings(rd_mol)
    ring_atoms = [a for ring in rings for a in ring]
    assert pool_atoms == set(ring_atoms)


def test_get_guest_atom_pool_all_heavy(eg5_ligands):

    rd_mol = eg5_ligands[0].to_rdkit()
    rmsf = np.zeros(rd_mol.GetNumAtoms()) * unit.nanometer
    rings = get_aromatic_rings(rd_mol)
    ring_atoms = [a for ring in rings for a in ring]
    # add high rmsf values for all ring atoms
    for i in ring_atoms:
        rmsf[i] = 1 * unit.nanometer

    pool_atoms, rings = _get_guest_atom_pool(
        rdmol=rd_mol,
        rmsf=rmsf,
        rmsf_cutoff=0.1 * unit.nanometer
    )
    # make sure only rings were found
    assert not rings
    # make sure we get heavy atoms with no rings
    heavy_atoms = get_heavy_atom_idxs(rd_mol)
    assert pool_atoms == set(heavy_atoms) - set(ring_atoms)


def test_get_guest_atom_pool_no_atoms(eg5_ligands):
    rd_mol = eg5_ligands[0].to_rdkit()
    rmsf = np.ones(rd_mol.GetNumAtoms()) * unit.nanometer
    pool_atoms, rings = _get_guest_atom_pool(
        rdmol=rd_mol,
        rmsf=rmsf,
        rmsf_cutoff=0.1 * unit.nanometer
    )
    # make sure only rings were found
    assert not rings
    # make sure no atoms are returned
    assert pool_atoms is None


def test_find_guest_atoms_normal(eg5_ligands):

    rd_mol = eg5_ligands[0].to_rdkit()
    lig = mda.Universe(rd_mol)
    angles = find_guest_atom_candidates(
        universe=lig,
        rdmol=rd_mol,
        guest_idxs=[a.ix for a in lig.atoms],
    )
    assert len(angles) == 24


def test_find_guest_atoms_no_atom_pool():
    with pytest.raises(ValueError, match="No suitable ligand atoms were found for the restraint"):
        lig = mda.Universe.from_smiles("CC")

        _ = find_guest_atom_candidates(
            universe=lig,
            rdmol=lig.atoms.convert_to("RDKIT"),
            guest_idxs=[a.ix for a in lig.atoms],
        )
