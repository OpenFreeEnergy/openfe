# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import MDAnalysis as mda
import pytest

from openfe.protocols.restraint_utils.geometry.boresch.geometry import (
    BoreschRestraintGeometry,
    find_boresch_restraint,
)

from openff.units import unit


@pytest.fixture()
def eg5_protein_ligand_universe(eg5_protein_pdb, eg5_ligands):
    protein = mda.Universe(eg5_protein_pdb)
    lig = mda.Universe(eg5_ligands[1].to_rdkit())
    # add the residue name of the ligand
    lig.add_TopologyAttr("resname", ["LIG"])
    return mda.Merge(protein.atoms, lig.atoms)


def test_get_boresh_missing_atoms(eg5_protein_ligand_universe, eg5_ligands):
    """
    Test an error is raised if we do not provide guest and host atoms
    """

    ligand_atoms = eg5_protein_ligand_universe.select_atoms("resname LIG")
    host_atoms = eg5_protein_ligand_universe.select_atoms("protein")
    with pytest.raises(ValueError, match="both ``guest_restraints_atoms_idxs`` and "):
        _ = find_boresch_restraint(
            universe=eg5_protein_ligand_universe,
            guest_rdmol=eg5_ligands[1].to_rdkit(),
            guest_idxs=[a.ix for a in ligand_atoms],
            host_idxs=[a.ix for a in host_atoms],
            host_selection="backbone",
            guest_restraint_atoms_idxs=[33, 12, 13]
        )


def test_boresh_too_few_host_atoms_found(eg5_protein_ligand_universe, eg5_ligands):
    """
    Test an error is raised if we can not find a set of host atoms
    """
    with pytest.raises(ValueError, match="Boresch-like restraint generation: too few atoms"):
        ligand_atoms = eg5_protein_ligand_universe.select_atoms("resname LIG")
        host_atoms = eg5_protein_ligand_universe.select_atoms("protein")
        _ = find_boresch_restraint(
            universe=eg5_protein_ligand_universe,
            guest_rdmol=eg5_ligands[1].to_rdkit(),
            guest_idxs=[a.ix for a in ligand_atoms],
            host_idxs=[a.ix for a in host_atoms],
            # select an atom group with no atoms
            host_selection="resnum 2",
        )


def test_boresh_restraint_user_defined(eg5_protein_ligand_universe, eg5_ligands):
    """
    Test creating a restraint with a user supplied set of atoms.
    """
    ligand_atoms = eg5_protein_ligand_universe.select_atoms("resname LIG")
    host_atoms = eg5_protein_ligand_universe.select_atoms("protein")
    restraint_geometry = find_boresch_restraint(
        universe=eg5_protein_ligand_universe,
        guest_rdmol=eg5_ligands[1].to_rdkit(),
        guest_idxs=[a.ix for a in ligand_atoms],
        host_idxs=[a.ix for a in host_atoms],
        host_selection="backbone",
        guest_restraint_atoms_idxs=[33, 12, 13],
        host_restraint_atoms_idxs=[3517, 1843, 3920]
    )
    host_ids = eg5_protein_ligand_universe.atoms[restraint_geometry.host_atoms]
    # make sure we have backbone atoms
    for a in host_ids:
        # backbone atom names
        assert a.name in ["CA", "C", "O", "N"]
    assert restraint_geometry.guest_atoms == [5528, 5507, 5508]
    assert restraint_geometry.host_atoms == [3517, 1843, 3920]
    # check the measured values
    assert 1.01590371 == pytest.approx(restraint_geometry.r_aA0.to("nanometer").m)
    assert 1.00327937 == pytest.approx(restraint_geometry.theta_A0.to("radians").m)
    assert 1.23561539 == pytest.approx(restraint_geometry.theta_B0.to("radians").m)
    assert 2.27961361 == pytest.approx(restraint_geometry.phi_A0.to("radians").m)
    assert 0.154240342 == pytest.approx(restraint_geometry.phi_B0.to("radians").m)
    assert -0.0239690127 == pytest.approx(restraint_geometry.phi_C0.to("radians").m)


def test_boresh_no_guest_atoms_found_ethane(eg5_protein_pdb):
    """
    Test an error is raised if we don't have enough ligand candidate atoms, we use ethane as
    it has no rings and less than 3 heavy atoms.
    """
    protein = mda.Universe(eg5_protein_pdb)
    # generate ethane with a single conformation
    lig = mda.Universe.from_smiles("CC")
    lig.add_TopologyAttr("resname", ["LIG"])
    universe = mda.Merge(protein.atoms, lig.atoms)

    ligand_atoms = universe.select_atoms("resname LIG")
    with pytest.raises(ValueError, match="No suitable ligand atoms were found for the restraint"):
        _ = find_boresch_restraint(
            universe=universe,
            guest_rdmol=lig.atoms.convert_to("RDKIT"),
            guest_idxs=[a.ix for a in ligand_atoms],
            host_idxs=[1, 2, 3],
            host_selection="backbone",
        )


def test_boresh_no_guest_atoms_found_collinear(eg5_protein_pdb):
    protein = mda.Universe(eg5_protein_pdb)
    # generate ethane with a single conformation
    lig = mda.Universe.from_smiles("O=C=O")
    lig.add_TopologyAttr("resname", ["LIG"])
    universe = mda.Merge(protein.atoms, lig.atoms)

    ligand_atoms = universe.select_atoms("resname LIG")
    with pytest.raises(ValueError, match="No suitable ligand atoms found for the restraint."):
        _ = find_boresch_restraint(
            universe=universe,
            guest_rdmol=lig.atoms.convert_to("RDKIT", force=True),
            guest_idxs=[a.ix for a in ligand_atoms],
            host_idxs=[1, 2, 3],
            host_selection="backbone",
        )


def test_boresh_no_host_atom_small_search(eg5_protein_ligand_universe, eg5_ligands):
    """
    Make sure an error is raised if no good host atom can be found by setting the search distance very low
    """
    ligand_atoms = eg5_protein_ligand_universe.select_atoms("resname LIG")
    host_atoms = eg5_protein_ligand_universe.select_atoms("protein")
    with pytest.raises(ValueError, match="No host atoms found within the search distance"):
        _ = find_boresch_restraint(
            universe=eg5_protein_ligand_universe,
            guest_rdmol=eg5_ligands[1].to_rdkit(),
            guest_idxs=[a.ix for a in ligand_atoms],
            host_idxs=[a.ix for a in host_atoms],
            host_selection="backbone",
            host_min_distance=0.0 * unit.angstrom,
            host_max_distance=0.1 * unit.angstrom
        )


def test_get_boresh_restraint_single_frame(eg5_protein_ligand_universe, eg5_ligands):
    """
    Make sure we can find a boresh restraint using a single frame
    """
    ligand_atoms = eg5_protein_ligand_universe.select_atoms("resname LIG")
    host_atoms = eg5_protein_ligand_universe.select_atoms("protein")
    restraint_geometry = find_boresch_restraint(
        universe=eg5_protein_ligand_universe,
        guest_rdmol=eg5_ligands[1].to_rdkit(),
        guest_idxs=[a.ix for a in ligand_atoms],
        host_idxs=[a.ix for a in host_atoms],
        host_selection="backbone",
    )
    assert isinstance(restraint_geometry, BoreschRestraintGeometry)
    host_ids = eg5_protein_ligand_universe.atoms[restraint_geometry.host_atoms]
    # make sure we have backbone atoms
    for a in host_ids:
        # backbone atom names
        assert a.name in ["CA", "C", "O", "N"]
    assert restraint_geometry.guest_atoms == [5528, 5507, 5508]
    assert restraint_geometry.host_atoms == [3517, 1843, 3920]
    # check the measured values
    assert 1.01590371 == pytest.approx(restraint_geometry.r_aA0.to("nanometer").m)
    assert 1.00327937 == pytest.approx(restraint_geometry.theta_A0.to("radians").m)
    assert 1.23561539 == pytest.approx(restraint_geometry.theta_B0.to("radians").m)
    assert 2.27961361 == pytest.approx(restraint_geometry.phi_A0.to("radians").m)
    assert 0.154240342 == pytest.approx(restraint_geometry.phi_B0.to("radians").m)
    assert -0.0239690127 == pytest.approx(restraint_geometry.phi_C0.to("radians").m)


def test_get_boresh_restraint_dssp(eg5_protein_ligand_universe, eg5_ligands):
    """
    Make sure we can find a boresh restraint using a single frame
    """
    ligand_atoms = eg5_protein_ligand_universe.select_atoms("resname LIG")
    host_atoms = eg5_protein_ligand_universe.select_atoms("protein")
    restraint_geometry = find_boresch_restraint(
        universe=eg5_protein_ligand_universe,
        guest_rdmol=eg5_ligands[1].to_rdkit(),
        guest_idxs=[a.ix for a in ligand_atoms],
        host_idxs=[a.ix for a in host_atoms],
        host_selection="backbone",
        dssp_filter=True
    )
    assert isinstance(restraint_geometry, BoreschRestraintGeometry)
    host_ids = eg5_protein_ligand_universe.atoms[restraint_geometry.host_atoms]
    # make sure we have backbone atoms
    for a in host_ids:
        # backbone atom names
        assert a.name in ["CA", "C", "O", "N"]
    # we should get the same guest atoms
    assert restraint_geometry.guest_atoms == [5528, 5507, 5508]
    # different host atoms
    assert restraint_geometry.host_atoms == [3058, 1895, 1933]
    # check the measured values
    assert 1.09084815 == pytest.approx(restraint_geometry.r_aA0.to("nanometer").m)
    assert 0.93824134 == pytest.approx(restraint_geometry.theta_A0.to("radians").m)
    assert 1.58227158 == pytest.approx(restraint_geometry.theta_B0.to("radians").m)
    assert 2.09875582 == pytest.approx(restraint_geometry.phi_A0.to("radians").m)
    assert -0.142658924 == pytest.approx(restraint_geometry.phi_B0.to("radians").m)
    assert -1.5340895 == pytest.approx(restraint_geometry.phi_C0.to("radians").m)
