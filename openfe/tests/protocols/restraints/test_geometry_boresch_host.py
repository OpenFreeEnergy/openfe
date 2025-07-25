# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import MDAnalysis as mda
import numpy as np
import pytest
from openfe.protocols.restraint_utils.geometry.boresch.host import (
    EvaluateHostAtoms1,
    EvaluateHostAtoms2,
    find_host_anchor,
    find_host_atom_candidates,
)
from openff.units import unit


@pytest.fixture()
def eg5_protein_ligand_universe(eg5_protein_pdb, eg5_ligands):
    protein = mda.Universe(eg5_protein_pdb)
    lig = mda.Universe(eg5_ligands[1].to_rdkit())
    # add the residue name of the ligand
    lig.add_TopologyAttr("resname", ["LIG"])
    return mda.Merge(protein.atoms, lig.atoms)


def test_host_atom_candidates_dssp(eg5_protein_ligand_universe):
    """
    Make sure both dssp warnings are triggered
    """

    host_atoms = eg5_protein_ligand_universe.select_atoms("protein")
    with (
        pytest.warns(
            match="Too few atoms found via secondary structure filtering will"
        ),
        pytest.warns(match="Too few atoms found in protein residue chains,"),
    ):
        _ = find_host_atom_candidates(
            universe=eg5_protein_ligand_universe,
            host_idxs=[a.ix for a in host_atoms],
            # hand picked
            l1_idx=5508,
            host_selection="backbone and resnum 15:25",
            dssp_filter=True,
        )


def test_host_atom_candidate_small_search(eg5_protein_ligand_universe):

    host_atoms = eg5_protein_ligand_universe.select_atoms("protein")
    with pytest.raises(
        ValueError, match="No host atoms found within the search distance"
    ):
        _ = find_host_atom_candidates(
            universe=eg5_protein_ligand_universe,
            host_idxs=[a.ix for a in host_atoms],
            # hand picked
            l1_idx=5508,
            host_selection="backbone",
            dssp_filter=False,
            max_distance=0.1 * unit.angstrom,
        )


def test_evaluate_host1_bad_ref(eg5_protein_ligand_universe):

    with pytest.raises(ValueError, match="Incorrect number of reference atoms passed"):
        _ = EvaluateHostAtoms1(
            reference=eg5_protein_ligand_universe.atoms,
            host_atom_pool=eg5_protein_ligand_universe.atoms,
            angle_force_constant=83.68 * unit.kilojoule_per_mole / unit.radians**2,
            temperature=298.15 * unit.kelvin,
            minimum_distance=1 * unit.nanometer,
        )


def test_evaluate_host1_good(eg5_protein_ligand_universe):

    angle_fc = 83.68 * unit.kilojoule_per_mole / unit.radians**2
    min_distance = 1 * unit.nanometer
    temp = 298.15 * unit.kelvin
    ho_eval = EvaluateHostAtoms1(
        # picked from a successful boresch restraint search
        reference=eg5_protein_ligand_universe.atoms[[5528, 5507, 5508]],
        host_atom_pool=eg5_protein_ligand_universe.select_atoms(
            "backbone and resnum 239"
        ),
        minimum_distance=min_distance,
        angle_force_constant=angle_fc,
        temperature=temp,
    )
    # make sure properties are used during the evaluation
    assert ho_eval.minimum_distance == min_distance.to("angstrom").m
    assert ho_eval.temperature == temp
    assert ho_eval.angle_force_constant == angle_fc
    ho_eval.run()
    # make sure all atoms in this residue are valid as this is the residue selected
    # during the automated search
    assert ho_eval.results.valid.all()
    assert not ho_eval.results.collinear.all()
    assert np.allclose(
        ho_eval.results.distances,
        np.array([[10.79778922], [10.15903706], [11.19430463], [11.36472103]]),
    )
    assert np.allclose(
        ho_eval.results.angles,
        np.array([[1.26279048], [1.23561539], [1.15134184], [1.04697413]]),
    )
    assert np.allclose(
        ho_eval.results.dihedrals,
        np.array([[0.10499465], [-0.02396901], [-0.09271532], [-0.06136335]]),
    )


def test_evaluate_host2_good(eg5_protein_ligand_universe):

    h2_eval = EvaluateHostAtoms2(
        # picked from a successful boresch restraint search
        reference=eg5_protein_ligand_universe.atoms[[5528, 5507, 5508]],
        host_atom_pool=eg5_protein_ligand_universe.select_atoms(
            "backbone and resnum 264"
        ),
        minimum_distance=1 * unit.nanometer,
        angle_force_constant=83.68 * unit.kilojoule_per_mole / unit.radians**2,
        temperature=298.15 * unit.kelvin,
    )
    h2_eval.run()
    # make sure all atoms in this residue are valid as this is the residue selected
    # during the automated search
    assert h2_eval.results.valid.all()
    assert not h2_eval.results.collinear.all()
    assert np.allclose(
        h2_eval.results.distances1,
        np.array([[12.91959211], [13.2744748], [12.9710364], [13.44522909]]),
    )
    assert np.allclose(
        h2_eval.results.distances2,
        np.array([[12.2098888], [12.68587248], [12.38582154], [12.77150153]]),
    )
    assert np.allclose(
        h2_eval.results.dihedrals,
        np.array([[0.4069051], [0.46465918], [0.59372385], [0.65580398]]),
    )


@pytest.mark.slow
def test_find_host_anchor_none(eg5_protein_ligand_universe):

    host_anchor = find_host_anchor(
        guest_atoms=eg5_protein_ligand_universe.atoms[[5528, 5507, 5508]],
        host_atom_pool=eg5_protein_ligand_universe.select_atoms("backbone"),
        minimum_distance=4.5 * unit.nanometer,
        angle_force_constant=83.68 * unit.kilojoule_per_mole / unit.radians**2,
        temperature=298.15 * unit.kelvin,
    )
    # we should get None if no atoms can be found
    assert host_anchor is None
