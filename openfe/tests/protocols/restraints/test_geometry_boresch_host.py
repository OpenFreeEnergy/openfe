# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import MDAnalysis as mda
import numpy as np
from numpy.testing import assert_equal
import os
import pathlib
import pytest
import pooch
from openfe.protocols.restraint_utils.geometry.boresch.host import (
    EvaluateHostAtoms1,
    EvaluateHostAtoms2,
    EvaluateBoreschAtoms,
    find_host_anchor_multi,
    find_host_anchor_bonded,
    find_host_atom_candidates,
)
from openfe.protocols.restraint_utils.geometry.utils import (
    is_collinear,
    check_angle_not_flat,
    check_dihedral_bounds,
)
from openff.units import unit

from ...conftest import HAS_INTERNET


POOCH_CACHE = pooch.os_cache("openfe")
zenodo_restraint_data = pooch.create(
    path=POOCH_CACHE,
    base_url="doi:10.5281/zenodo.15212342",
    registry={
        "t4_lysozyme_trajectory.zip": "sha256:e985d055db25b5468491e169948f641833a5fbb67a23dbb0a00b57fb7c0e59c8"
    },
    retry_if_failed=5,
)


@pytest.fixture(scope='module')
def t4_lysozyme_trajectory_universe():
    zenodo_restraint_data.fetch("t4_lysozyme_trajectory.zip", processor=pooch.Unzip())
    cache_dir = pathlib.Path(
        pooch.os_cache("openfe")
        / "t4_lysozyme_trajectory.zip.unzip/t4_lysozyme_trajectory"
    )
    universe = mda.Universe(
        str(cache_dir / "t4_toluene_complex.pdb"),
        str(cache_dir / "t4_toluene_complex.xtc"),
    )
    # guess bonds for the protein atoms
    universe.select_atoms('protein').guess_bonds()
    return universe

@pytest.fixture
def eg5_protein_ligand_universe(eg5_protein_pdb, eg5_ligands):
    protein = mda.Universe(eg5_protein_pdb)
    lig = mda.Universe(eg5_ligands[1].to_rdkit())
    # add the residue name of the ligand
    lig.add_TopologyAttr("resname", ["LIG"])
    return mda.Merge(protein.atoms, lig.atoms)


@pytest.fixture
def eg5_protein_ligand_universe_bonded(eg5_protein_pdb, eg5_ligands):
    protein = mda.Universe(eg5_protein_pdb)
    lig = mda.Universe(eg5_ligands[1].to_rdkit())
    # add the residue name of the ligand
    lig.add_TopologyAttr("resname", ["LIG"])
    merged_u = mda.Merge(protein.atoms, lig.atoms)
    merged_u.guess_TopologyAttrs(context='default', to_guess=['bonds', 'angles'])
    return merged_u


def test_host_atom_candidates_dssp(eg5_protein_ligand_universe):
    """
    Run DSSP search normally
    """
    host_atoms = eg5_protein_ligand_universe.select_atoms("protein")

    idxs = find_host_atom_candidates(
        universe=eg5_protein_ligand_universe,
        host_idxs=host_atoms.ix,
        # hand picked
        guest_anchor_idx=5508,
        host_selection="backbone and resnum 212:221",
        dssp_filter=True,
    )
    expected = np.array(
        [3144, 3146, 3145, 3143, 3162, 3206, 3200, 3207, 3126, 3201, 3127,
         3163, 3199, 3202, 3164, 3125, 3165, 3177, 3208, 3179, 3124, 3216,
         3209, 3109, 3107, 3178, 3110, 3180, 3108, 3248, 3217, 3249, 3226,
         3218, 3228, 3227, 3250, 3219, 3251, 3229]
    )
    assert_equal(idxs, expected)


def test_host_atom_candidates_dssp_too_few_atoms(eg5_protein_ligand_universe):
    """
    Make sure both dssp warnings are triggered
    """

    host_atoms = eg5_protein_ligand_universe.select_atoms("protein")
    with (
        pytest.warns(match="DSSP filter found"),
        pytest.warns(match="protein chain filter found"),
    ):
        _ = find_host_atom_candidates(
            universe=eg5_protein_ligand_universe,
            host_idxs=host_atoms.ix,
            # hand picked
            guest_anchor_idx=5508,
            host_selection="backbone and resnum 15:25",
            dssp_filter=True,
            max_search_distance=2*unit.nanometer
        )


def test_host_atom_candidate_small_search(eg5_protein_ligand_universe):

    host_atoms = eg5_protein_ligand_universe.select_atoms("protein")
    with pytest.raises(
        ValueError, match="No host atoms found within the search distance"
    ):
        _ = find_host_atom_candidates(
            universe=eg5_protein_ligand_universe,
            host_idxs=host_atoms.ix,
            # hand picked
            guest_anchor_idx=5508,
            host_selection="backbone",
            dssp_filter=False,
            max_search_distance=0.1 * unit.angstrom,
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
class TestFindAnchorMulti:
    # G0-G1-G2
    guest_atoms = [5528, 5507, 5508]
    ref_anchor_indices = [133, 1, 16]
    ref_h0g0_distance = 20.612924
    ref_h0h1_distance = 25.805103
    ref_h0h2_distance = 19.68613
    ref_h1h2_distance = 7.47768

    @pytest.fixture
    def universe(self, eg5_protein_ligand_universe):
        return eg5_protein_ligand_universe

    @pytest.fixture
    def host_anchor(self, universe):
        return find_host_anchor_multi(
            guest_atoms=universe.atoms[self.guest_atoms],
            host_atom_pool=universe.select_atoms("backbone"),
            host_minimum_distance=0.5 * unit.nanometer,
            guest_minimum_distance=2 * unit.nanometer,
            angle_force_constant=83.68 * unit.kilojoule_per_mole / unit.radians**2,
            temperature=298.15 * unit.kelvin,
        )

    def test_anchor_regression(self, host_anchor):
        # regression test the anchor we find
        assert_equal(host_anchor, self.ref_anchor_indices)

    def test_host_guest_bond_distance(self, host_anchor, universe):
        # check that the l0 g0 distance is at least the `guest_minimum_distance` (2nm)  for the `host_anchor` fixture.
        dist = mda.lib.distances.calc_bonds(
            universe.atoms[host_anchor[0]].position,
            universe.atoms[self.guest_atoms[0]].position,
            box=universe.dimensions,
        )

        assert dist == pytest.approx(self.ref_h0g0_distance, abs=1e-5)

    def test_host_distances(self, host_anchor, universe):
        # check the h0-h1, h1-h2, and h0-h2 distances
        for i, j, ref in [
            [0, 1, self.ref_h0h1_distance],
            [1, 2, self.ref_h1h2_distance],
            [0, 2, self.ref_h0h2_distance]
        ]:
            dist = mda.lib.distances.calc_bonds(
                universe.atoms[host_anchor[i]].position,
                universe.atoms[host_anchor[j]].position,
                box=universe.dimensions,
            )
            assert dist == pytest.approx(ref, abs=1e-5)

    def test_not_collinear(self, host_anchor, universe):
        # check none of the g2-g1-g0-h0-h1-h2 vectors are not collinear
        assert not is_collinear(
            positions=np.vstack((
                universe.atoms[self.guest_atoms[::-1]].positions,
                universe.atoms[host_anchor].positions
            )),
            atoms=[0, 1, 2, 3, 4, 5],
            dimensions=universe.dimensions
        )

    def test_angles(self, host_anchor, universe):
        # check that the angles aren't flat
        ag1 = mda.lib.distances.calc_angles(
            universe.atoms[self.guest_atoms[1]].position,
            universe.atoms[self.guest_atoms[0]].position,
            universe.atoms[host_anchor[0]].position,
            box=universe.dimensions
        )
        ag2 = mda.lib.distances.calc_angles(
            universe.atoms[self.guest_atoms[0]].position,
            universe.atoms[host_anchor[0]].position,
            universe.atoms[host_anchor[1]].position,
            box=universe.dimensions
        )
        for angle in [ag1, ag2]:
            assert check_angle_not_flat(
                angle=angle * unit.radians,
                force_constant=83.68 * unit.kilojoule_per_mole / unit.radians**2,
                temperature=298.15 * unit.kelvin,
            )

    def test_dihedrals(self, host_anchor, universe):
        dih1 = mda.lib.distances.calc_dihedrals(
            universe.atoms[self.guest_atoms[2]].position,
            universe.atoms[self.guest_atoms[1]].position,
            universe.atoms[self.guest_atoms[0]].position,
            universe.atoms[host_anchor[0]].position,
            box=universe.dimensions,
        )
        dih2 = mda.lib.distances.calc_dihedrals(
            universe.atoms[self.guest_atoms[1]].position,
            universe.atoms[self.guest_atoms[0]].position,
            universe.atoms[host_anchor[0]].position,
            universe.atoms[host_anchor[1]].position,
            box=universe.dimensions,
        )
        dih3 = mda.lib.distances.calc_dihedrals(
            universe.atoms[self.guest_atoms[0]].position,
            universe.atoms[host_anchor[0]].position,
            universe.atoms[host_anchor[1]].position,
            universe.atoms[host_anchor[2]].position,
            box=universe.dimensions,
        )
        assert check_dihedral_bounds(dih1 * unit.radians)
        assert check_dihedral_bounds(dih2 * unit.radians)
        assert check_dihedral_bounds(dih3 * unit.radians)


@pytest.mark.slow
class TestFindAnchorBonded(TestFindAnchorMulti):
    ref_anchor_indices = [133, 119, 118]
    ref_h0g0_distance = 20.612924
    ref_h0h1_distance = 1.34244
    ref_h0h2_distance = 2.44758
    ref_h1h2_distance = 1.53359

    @pytest.fixture
    def universe(self, eg5_protein_ligand_universe_bonded):
        return eg5_protein_ligand_universe_bonded

    @pytest.fixture
    def host_anchor(self, universe):
        return find_host_anchor_bonded(
            guest_atoms=universe.atoms[self.guest_atoms],
            host_atom_pool=universe.select_atoms("backbone"),
            guest_minimum_distance=0.5 * unit.nanometer,
            angle_force_constant=83.68 * unit.kilojoule_per_mole / unit.radians**2,
            temperature=298.15 * unit.kelvin,
        )


@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.exists(POOCH_CACHE) and not HAS_INTERNET,
    reason="Internet seems to be unavailable and test data is not cached locally.",
)
class TestFindAnchorBondedTrajectory(TestFindAnchorMulti):
    guest_atoms = [2611, 2612, 2613]
    ref_anchor_indices = [1253, 1254, 1255]
    ref_h0g0_distance = 13.08494
    ref_h0h1_distance = 1.50217
    ref_h0h2_distance = 2.61515
    ref_h1h2_distance = 1.55881

    @pytest.fixture(scope='class')
    def universe(self, t4_lysozyme_trajectory_universe):
        return t4_lysozyme_trajectory_universe

    @pytest.fixture(scope='class')
    def host_anchor(self, universe):
        return find_host_anchor_bonded(
            guest_atoms=universe.atoms[self.guest_atoms],
            host_atom_pool=universe.select_atoms("backbone"),
            guest_minimum_distance=0.0 * unit.nanometer,
            angle_force_constant=83.68 * unit.kilojoule_per_mole / unit.radians**2,
            temperature=298.15 * unit.kelvin,
        )


def test_boresch_evaluation_noatomgroup_error():
    errmsg = "Need to have at least one restraint"
    with pytest.raises(ValueError, match=errmsg):
        EvaluateBoreschAtoms(
            restraints=[],
            angle_force_constant=83.68 * unit.kilojoule_per_mole / unit.radians**2,
            temperature=298.15 * unit.kelvin,
        )


def test_boresch_evaluation_incorrectnumber_error(eg5_protein_ligand_universe):
    ag = eg5_protein_ligand_universe.atoms[:4]
    errmsg = "Incorrect number of restraint atoms passed"
    with pytest.raises(ValueError, match=errmsg):
        EvaluateBoreschAtoms(
            restraints=[ag],
            angle_force_constant=83.68 * unit.kilojoule_per_mole / unit.radians**2,
            temperature=298.15 * unit.kelvin,
        )


@pytest.mark.slow
def test_find_host_anchor_multi_none(eg5_protein_ligand_universe):

    host_anchor = find_host_anchor_multi(
        guest_atoms=eg5_protein_ligand_universe.atoms[[5528, 5507, 5508]],
        host_atom_pool=eg5_protein_ligand_universe.select_atoms("backbone"),
        # Setting host and guest minimum distances to a large value so
        # we find no atoms.
        host_minimum_distance=4.5 * unit.nanometer,
        guest_minimum_distance=4.5 * unit.nanometer,
        angle_force_constant=83.68 * unit.kilojoule_per_mole / unit.radians**2,
        temperature=298.15 * unit.kelvin,
    )
    # we should get None if no atoms can be found
    assert host_anchor is None


@pytest.mark.slow
def test_find_host_anchor_bonded_none(eg5_protein_ligand_universe_bonded):

    host_anchor = find_host_anchor_bonded(
        guest_atoms=eg5_protein_ligand_universe_bonded.atoms[[5528, 5507, 5508]],
        host_atom_pool=eg5_protein_ligand_universe_bonded.select_atoms("backbone"),
        # Setting host and guest minimum distances to a large value so
        # we find no atoms.
        guest_minimum_distance=4.5 * unit.nanometer,
        angle_force_constant=83.68 * unit.kilojoule_per_mole / unit.radians**2,
        temperature=298.15 * unit.kelvin,
    )

    # we should get None if no atoms can be found
    assert host_anchor is None


@pytest.mark.slow
def test_find_host_anchor_bonded_nobonds_none(eg5_protein_ligand_universe):

    # No angles were found, so it will attempt to find some
    # It'll fail because there are no bonds available.
    with pytest.warns(UserWarning, match="no angles found"):
        host_anchor = find_host_anchor_bonded(
            guest_atoms=eg5_protein_ligand_universe.atoms[[5528, 5507, 5508]],
            host_atom_pool=eg5_protein_ligand_universe.select_atoms("backbone"),
            # Setting host and guest minimum distances to a large value so
            # we find no atoms.
            guest_minimum_distance=0.5 * unit.nanometer,
            angle_force_constant=83.68 * unit.kilojoule_per_mole / unit.radians**2,
            temperature=298.15 * unit.kelvin,
        )

    # we should get None if no atoms can be found
    assert host_anchor is None
