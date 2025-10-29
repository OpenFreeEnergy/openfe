# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import MDAnalysis as mda
import pytest
from openfe.protocols.restraint_utils.geometry.harmonic import (
    DistanceRestraintGeometry,
    get_distance_restraint,
    get_molecule_centers_restraint,
)


@pytest.fixture()
def eg5_protein_ligand_universe(eg5_protein_pdb, eg5_ligands):
    protein = mda.Universe(eg5_protein_pdb)
    lig = mda.Universe(eg5_ligands[1].to_rdkit())
    # add the residue name of the ligand
    lig.add_TopologyAttr("resname", ["LIG"])
    return mda.Merge(protein.atoms, lig.atoms)


def test_hostguest_geometry():
    """
    A very basic will it build test.
    """
    geom = DistanceRestraintGeometry(guest_atoms=[1, 2, 3], host_atoms=[4])

    assert isinstance(geom, DistanceRestraintGeometry)


def test_get_distance_restraint_selection(eg5_protein_ligand_universe):
    """
    Check that you get a distance restraint using atom selections.
    """
    expected_guest_atoms = eg5_protein_ligand_universe.select_atoms("resname LIG and not name H*")
    water_atoms = eg5_protein_ligand_universe.select_atoms("resname HOH")
    restraint_geometry = get_distance_restraint(
        universe=eg5_protein_ligand_universe,
        host_selection="backbone and same resid as (around 4 resname LIG) and not resname HOH",
        guest_selection="resname LIG and not name H*",
    )

    # make sure the guest atoms cover all heavy atoms in the ligand
    assert restraint_geometry.guest_atoms == [a.ix for a in expected_guest_atoms]
    # make sure no water atoms are selected as a host
    assert not any(a.ix for a in water_atoms if a.ix in restraint_geometry.host_atoms)
    assert isinstance(restraint_geometry, DistanceRestraintGeometry)


def test_get_distance_restraint_atom_list(eg5_protein_ligand_universe):
    """
    Check that we can get a restraint using a set of host and guest atom lists
    """
    expected_guest_atoms = eg5_protein_ligand_universe.select_atoms("resname LIG and not name H*")
    host_atoms = [1, 2, 3]
    restraint_geometry = get_distance_restraint(
        universe=eg5_protein_ligand_universe,
        # take the first few protein atoms
        host_atoms=host_atoms,
        # take the first few ligand atoms
        guest_atoms=[5496, 5497, 5498, 5500],
    )
    guest_atoms = [a.ix for a in expected_guest_atoms]
    assert all(i in guest_atoms for i in restraint_geometry.guest_atoms)
    assert restraint_geometry.host_atoms == host_atoms


def test_get_molecule_centers_restraint(eg5_ligands):
    """
    Create a centers distance restraint between pairs of ligands
    """
    ligand_a, ligand_b = eg5_ligands
    lig_a_rdmol = ligand_a.to_rdkit()
    n_atoms_a = lig_a_rdmol.GetNumAtoms()
    lig_b_rdmol = ligand_b.to_rdkit()
    restraint_geometry = get_molecule_centers_restraint(
        molA_rdmol=lig_a_rdmol,
        molB_rdmol=lig_b_rdmol,
        molA_idxs=[i for i in range(n_atoms_a)],
        molB_idxs=[i + n_atoms_a for i in range(lig_b_rdmol.GetNumAtoms())],
    )
    assert isinstance(restraint_geometry, DistanceRestraintGeometry)
    assert restraint_geometry.guest_atoms[0] <= n_atoms_a
    assert restraint_geometry.host_atoms[0] > n_atoms_a
