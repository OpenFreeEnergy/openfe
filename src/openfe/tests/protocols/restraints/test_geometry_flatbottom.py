# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import MDAnalysis as mda
import pytest
from openff.units import unit

from openfe.protocols.restraint_utils.geometry.flatbottom import (
    FlatBottomDistanceGeometry,
    get_flatbottom_distance_restraint,
)


@pytest.fixture()
def eg5_protein_ligand_universe(eg5_protein_pdb, eg5_ligands):
    protein = mda.Universe(eg5_protein_pdb)
    lig = mda.Universe(eg5_ligands[1].to_rdkit())
    # add the residue name of the ligand
    lig.add_TopologyAttr("resname", ["LIG"])
    return mda.Merge(protein.atoms, lig.atoms)


def test_no_atoms_found(eg5_protein_ligand_universe):
    with pytest.raises(ValueError, match="no atoms found in either the host or guest"):
        _ = get_flatbottom_distance_restraint(
            universe=eg5_protein_ligand_universe,
            # the protein starts at 15
            host_selection="resnum 2",
            # the ligand is resnum 1, get only the heavy atoms
            guest_selection="resname LIG and not name H*",
        )


@pytest.mark.parametrize(
    "padding, well_radius",
    [
        pytest.param(0.5, 0.666, id="0.5"),
        pytest.param(0.8, 0.966, id="0.8"),
    ],
)
def test_get_flatbottom_restraint_from_selection(eg5_protein_ligand_universe, padding, well_radius):
    expected_guest_atoms = eg5_protein_ligand_universe.select_atoms("resname LIG and not name H*")
    water_atoms = eg5_protein_ligand_universe.select_atoms("resname HOH")
    restraint_geometry = get_flatbottom_distance_restraint(
        universe=eg5_protein_ligand_universe,
        # select all residues around the ligand
        host_selection="backbone and same resid as (around 4 resname LIG) and not resname HOH",
        # get all heavy atoms in the ligand
        guest_selection="resname LIG and not name H*",
        padding=padding * unit.nanometer,
    )
    # make sure the guest atoms cover all heavy atoms in the ligand
    assert restraint_geometry.guest_atoms == [a.ix for a in expected_guest_atoms]
    # make sure no water atoms are selected as a host
    assert not any(a.ix for a in water_atoms if a.ix in restraint_geometry.host_atoms)
    assert isinstance(restraint_geometry, FlatBottomDistanceGeometry)
    # probably could have a tighter check if we wanted
    assert well_radius == pytest.approx(restraint_geometry.well_radius.to("nanometer").m, abs=1e-4)


def test_get_flatbottom_restraint_from_atoms(eg5_protein_ligand_universe):
    expected_guest_atoms = eg5_protein_ligand_universe.select_atoms("resname LIG and not name H*")
    host_atoms = eg5_protein_ligand_universe.select_atoms(
        "backbone and same resid as (around 4 resname LIG) and not resname HOH"
    )
    host_atom_ix = [a.ix for a in host_atoms]
    restraint_geometry = get_flatbottom_distance_restraint(
        universe=eg5_protein_ligand_universe,
        # take all host atoms within 4 angstroms
        host_atoms=host_atom_ix,
        # take the first few ligand atoms
        guest_atoms=[5496, 5497, 5498, 5500],
    )
    guest_atoms = [a.ix for a in expected_guest_atoms]
    assert all(i in guest_atoms for i in restraint_geometry.guest_atoms)
    assert restraint_geometry.host_atoms == host_atom_ix
    assert 1.1415 == pytest.approx(restraint_geometry.well_radius.to("nanometer").m, abs=1e-4)
