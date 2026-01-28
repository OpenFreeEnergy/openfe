# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import os
import pathlib

import MDAnalysis as mda
import pooch
import pytest
from openff.units import unit
from rdkit import Chem

from openfe.data._registry import POOCH_CACHE, zenodo_industry_benchmark_systems
from openfe.protocols.restraint_utils.geometry.boresch.geometry import (
    BoreschRestraintGeometry,
    find_boresch_restraint,
)

from ...conftest import HAS_INTERNET


@pytest.fixture()
def eg5_protein_ligand_universe(eg5_protein_pdb, eg5_ligands):
    protein = mda.Universe(eg5_protein_pdb)
    lig = mda.Universe(eg5_ligands[1].to_rdkit())
    # add the residue name of the ligand
    lig.add_TopologyAttr("resname", ["LIG"])
    return mda.Merge(protein.atoms, lig.atoms)


def test_get_boresch_missing_atoms(eg5_protein_ligand_universe, eg5_ligands):
    """
    Test an error is raised if we do not provide guest and host atoms
    """

    ligand_atoms = eg5_protein_ligand_universe.select_atoms("resname LIG")
    host_atoms = eg5_protein_ligand_universe.select_atoms("protein")
    with pytest.raises(ValueError, match="both ``guest_restraints_atoms_idxs`` and "):
        _ = find_boresch_restraint(
            universe=eg5_protein_ligand_universe,
            guest_rdmol=eg5_ligands[1].to_rdkit(),
            guest_idxs=ligand_atoms.atoms.ix,
            host_idxs=host_atoms.atoms.ix,
            host_selection="backbone",
            guest_restraint_atoms_idxs=[33, 12, 13],
        )


def test_boresch_too_few_host_atoms_found(eg5_protein_ligand_universe, eg5_ligands):
    """
    Test an error is raised if we can not find a set of host atoms
    """
    with pytest.raises(ValueError, match="Boresch-like restraint generation: too few atoms"):
        ligand_atoms = eg5_protein_ligand_universe.select_atoms("resname LIG")
        host_atoms = eg5_protein_ligand_universe.select_atoms("protein")
        _ = find_boresch_restraint(
            universe=eg5_protein_ligand_universe,
            guest_rdmol=eg5_ligands[1].to_rdkit(),
            guest_idxs=ligand_atoms.atoms.ix,
            host_idxs=host_atoms.atoms.ix,
            # select an atom group with no atoms
            host_selection="resnum 2",
        )


def test_boresch_restraint_user_defined(eg5_protein_ligand_universe, eg5_ligands):
    """
    Test creating a restraint with a user supplied set of atoms.
    """
    ligand_atoms = eg5_protein_ligand_universe.select_atoms("resname LIG")
    host_atoms = eg5_protein_ligand_universe.select_atoms("protein")
    restraint_geometry = find_boresch_restraint(
        universe=eg5_protein_ligand_universe,
        guest_rdmol=eg5_ligands[1].to_rdkit(),
        guest_idxs=ligand_atoms.atoms.ix,
        host_idxs=host_atoms.atoms.ix,
        host_selection="backbone",
        guest_restraint_atoms_idxs=[33, 12, 13],
        host_restraint_atoms_idxs=[3517, 1843, 3920],
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


def test_boresch_no_guest_atoms_found_ethane(eg5_protein_pdb):
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
            guest_idxs=ligand_atoms.atoms.ix,
            host_idxs=[1, 2, 3],
            host_selection="backbone",
        )


def test_boresch_no_guest_atoms_found_collinear(eg5_protein_pdb):
    protein = mda.Universe(eg5_protein_pdb)
    # generate carbondioxide with a single conformation
    lig = mda.Universe.from_smiles("O=C=O")
    lig.add_TopologyAttr("resname", ["LIG"])
    universe = mda.Merge(protein.atoms, lig.atoms)

    ligand_atoms = universe.select_atoms("resname LIG")
    with pytest.raises(ValueError, match="No suitable ligand atoms found for the restraint."):
        _ = find_boresch_restraint(
            universe=universe,
            guest_rdmol=lig.atoms.convert_to("RDKIT", force=True),
            guest_idxs=ligand_atoms.atoms.ix,
            host_idxs=[1, 2, 3],
            host_selection="backbone",
        )


def test_boresch_no_host_atom_pool(eg5_protein_ligand_universe, eg5_ligands):
    """
    Make sure an error is raised if no good host atom can be found by setting the search distance very low
    """
    ligand_atoms = eg5_protein_ligand_universe.select_atoms("resname LIG")
    host_atoms = eg5_protein_ligand_universe.select_atoms("protein")
    with pytest.raises(ValueError, match="No host atoms found within the search distance"):
        _ = find_boresch_restraint(
            universe=eg5_protein_ligand_universe,
            guest_rdmol=eg5_ligands[1].to_rdkit(),
            guest_idxs=ligand_atoms.atoms.ix,
            host_idxs=host_atoms.atoms.ix,
            host_selection="backbone",
            host_min_distance=0.0 * unit.angstrom,
            host_max_distance=0.1 * unit.angstrom,
        )


def test_boresch_no_host_anchor(eg5_protein_ligand_universe, eg5_ligands):
    """
    Make sure an error is raised if we can not find a host anchor from the pool.
    We limit the selection to a single TYR near the binding site to
    force only one residue being picked up.
    """

    ligand_atoms = eg5_protein_ligand_universe.select_atoms("resname LIG")
    host_atoms = eg5_protein_ligand_universe.select_atoms("protein")
    with pytest.raises(ValueError, match="No suitable host atoms could be found"):
        _ = find_boresch_restraint(
            universe=eg5_protein_ligand_universe,
            guest_rdmol=eg5_ligands[1].to_rdkit(),
            guest_idxs=ligand_atoms.atoms.ix,
            host_idxs=host_atoms.atoms.ix,
            host_selection="backbone and resname TYR",
            host_min_distance=0 * unit.nanometers,
            host_max_distance=1 * unit.nanometers,
        )


@pytest.mark.slow
def test_get_boresch_restraint_single_frame(eg5_protein_ligand_universe, eg5_ligands):
    """
    Make sure we can find a boresh restraint using a single frame
    """
    ligand_atoms = eg5_protein_ligand_universe.select_atoms("resname LIG")
    host_atoms = eg5_protein_ligand_universe.select_atoms("protein")
    restraint_geometry = find_boresch_restraint(
        universe=eg5_protein_ligand_universe,
        guest_rdmol=eg5_ligands[1].to_rdkit(),
        guest_idxs=ligand_atoms.atoms.ix,
        host_idxs=host_atoms.atoms.ix,
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


def test_get_boresch_restraint_dssp(eg5_protein_ligand_universe, eg5_ligands):
    """
    Make sure we can find a boresh restraint using a single frame
    """
    ligand_atoms = eg5_protein_ligand_universe.select_atoms("resname LIG")
    host_atoms = eg5_protein_ligand_universe.select_atoms("protein")
    restraint_geometry = find_boresch_restraint(
        universe=eg5_protein_ligand_universe,
        guest_rdmol=eg5_ligands[1].to_rdkit(),
        guest_idxs=ligand_atoms.atoms.ix,
        host_idxs=host_atoms.atoms.ix,
        host_selection="backbone",
        dssp_filter=True,
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
    assert restraint_geometry.host_atoms == [3517, 3058, 3548]
    # check the measured values
    assert 1.01590371 == pytest.approx(restraint_geometry.r_aA0.to("nanometer").m)
    assert 0.84966606 == pytest.approx(restraint_geometry.theta_A0.to("radians").m)
    assert 1.23561539 == pytest.approx(restraint_geometry.theta_B0.to("radians").m)
    assert 2.56825286 == pytest.approx(restraint_geometry.phi_A0.to("radians").m)
    assert -1.60162692 == pytest.approx(restraint_geometry.phi_B0.to("radians").m)
    assert -0.02396901 == pytest.approx(restraint_geometry.phi_C0.to("radians").m)


pooch_industry_benchmark_systems = pooch.create(
    path=POOCH_CACHE,
    base_url=zenodo_industry_benchmark_systems["base_url"],
    registry={
        zenodo_industry_benchmark_systems["fname"]: zenodo_industry_benchmark_systems["known_hash"]
    },
)


@pytest.fixture
def industry_benchmark_files():
    pooch_industry_benchmark_systems.fetch(
        "industry_benchmark_systems.zip", processor=pooch.Unzip()
    )
    cache_dir = pathlib.Path(
        pooch.os_cache("openfe") / "industry_benchmark_systems.zip.unzip/industry_benchmark_systems"
    )
    return cache_dir


@pytest.mark.skipif(
    not os.path.exists(POOCH_CACHE) and not HAS_INTERNET,
    reason="Internet seems to be unavailable and test data is not cached locally.",
)
@pytest.mark.parametrize(
    "system",
    [
        "jacs_set/bace",
        "jacs_set/cdk2",
        "jacs_set/jnk1",
        "jacs_set/mcl1",
        "jacs_set/p38",
        "jacs_set/ptp1b",
        "jacs_set/thrombin",
        "jacs_set/tyk2",
        "janssen_bace/bace_ciordia_prospective",
        "janssen_bace/bace_p3_arg368_in",
        "janssen_bace/ciordia_retro",
        "janssen_bace/keranen_p2",
        "merck/cdk8",
        "merck/cmet",
        "merck/eg5",
        "merck/hif2a",
        "merck/pfkfb3",
        "merck/shp2",
        "merck/syk",
        "merck/tnks2",
        "mcs_docking_set/hne",
        "mcs_docking_set/renin",
        "fragments/hsp90_2rings",
        "fragments/hsp90_single_ring",
        "fragments/jak2_set1",
        "fragments/jak2_set2",
        "fragments/liga",
        "fragments/mcl1",
        "fragments/mup1",
        "fragments/p38",
        "fragments/t4_lysozyme",
        "miscellaneous_set/btk",
        "miscellaneous_set/cdk8",
        "miscellaneous_set/faah",
        "miscellaneous_set/galectin",
        "miscellaneous_set/hiv1_protease",
    ],
)
def test_get_boresch_restrain_industry_benchmark_systems(system, industry_benchmark_files):
    """
    Regression test generating boresch restraints for a single frame for most industry benchmark systems.
    Currently, a single ligand is used from each system and the expected reference data is stored as SDtags
    on the ligand.
    """
    # load the protein
    protein = mda.Universe(str(industry_benchmark_files / system / "protein.pdb"))
    # load the ligand
    ligand = [
        m
        for m in Chem.SDMolSupplier(
            str(industry_benchmark_files / system / "test_ligand.sdf"), removeHs=False
        )
    ][0]
    lig_uni = mda.Universe(ligand)
    lig_uni.add_TopologyAttr("resname", ["LIG"])
    universe = mda.Merge(protein.atoms, lig_uni.atoms)

    ligand_atoms = universe.select_atoms("resname LIG")
    lig_ids = ligand_atoms.atoms.ix
    host_atoms = universe.select_atoms("protein")
    host_ids = host_atoms.atoms.ix

    # create the geometry
    restraint_geometry = find_boresch_restraint(
        universe=universe,
        guest_rdmol=ligand,
        guest_idxs=lig_ids,
        host_idxs=host_ids,
        host_selection="backbone",
        anchor_finding_strategy="multi-residue",
        dssp_filter=False,
        # reduce the search space for CI speed!
        host_max_distance=1.5 * unit.nanometer,
    )

    # make sure we have backbone atoms as requested
    host_restrain_atoms = universe.atoms[restraint_geometry.host_atoms]
    for a in host_restrain_atoms:
        # backbone atom names
        assert a.name in ["CA", "C", "O", "N"]

    # make sure the host/guest atoms are in the selection we gave
    assert all(i in lig_ids for i in restraint_geometry.guest_atoms)
    assert all(i in host_ids for i in restraint_geometry.host_atoms)

    # finally make sure we get the expected values
    for i, atom in enumerate(restraint_geometry.host_atoms):
        assert ligand.GetIntProp(f"Host{i}") == atom
    for i, atom in enumerate(restraint_geometry.guest_atoms):
        assert ligand.GetIntProp(f"Guest{i}") == atom
    for prop in ["r_aA0", "theta_A0", "theta_B0", "phi_A0", "phi_B0", "phi_C0"]:
        assert pytest.approx(ligand.GetDoubleProp(prop)) == getattr(restraint_geometry, prop).m
