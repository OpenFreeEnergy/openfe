# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import MDAnalysis as mda
import numpy as np
from numpy.testing import assert_equal
import pytest
from openfe.protocols.restraint_utils.geometry.boresch.host import (
    EvaluateHostAtoms1,
    EvaluateHostAtoms2,
    find_host_anchor,
    find_host_atom_candidates,
)
from openfe.protocols.restraint_utils.geometry.utils import (
    is_collinear,
    check_angle_not_flat,
    check_dihedral_bounds,
)
from openff.units import unit


@pytest.fixture
def eg5_protein_ligand_universe(eg5_protein_pdb, eg5_ligands):
    protein = mda.Universe(eg5_protein_pdb)
    lig = mda.Universe(eg5_ligands[1].to_rdkit())
    # add the residue name of the ligand
    lig.add_TopologyAttr("resname", ["LIG"])
    return mda.Merge(protein.atoms, lig.atoms)


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
class TestFindAnchor:
    @pytest.fixture
    def universe(self, eg5_protein_ligand_universe):
        return eg5_protein_ligand_universe

    @pytest.fixture
    def host_anchor(self, eg5_protein_ligand_universe):
        return find_host_anchor(
            guest_atoms=eg5_protein_ligand_universe.atoms[[5528, 5507, 5508]],
            host_atom_pool=eg5_protein_ligand_universe.select_atoms("backbone"),
            host_minimum_distance=0.5 * unit.nanometer,
            guest_minimum_distance=2 * unit.nanometer,
            angle_force_constant=83.68 * unit.kilojoule_per_mole / unit.radians**2,
            temperature=298.15 * unit.kelvin,
        )

    def test_anchor_regression(self, host_anchor):
        # regression test the anchor we find
        assert_equal(host_anchor, [133, 1, 16])

    def test_host_guest_bond_distance(self, host_anchor, universe):
        # check that the l0 g0 distance is >= 2 nm away
        dist = mda.lib.distances.calc_bonds(
            universe.atoms[host_anchor[0]].position,
            universe.atoms[5528].position,
            box=universe.dimensions,
        )

        # distance is just about 2.0 nm
        assert dist == pytest.approx(20.612924)

    def test_host_distances(self, host_anchor, universe):
        # check the h0-h1, h1-h2, and h0-h2 distances
        for i, j, ref in [[0, 1, 25.805103], [1, 2, 7.47768], [0, 2, 19.68613]]:
            dist = mda.lib.distances.calc_bonds(
                universe.atoms[host_anchor[i]].position,
                universe.atoms[host_anchor[j]].position,
                box=universe.dimensions,
            )
            assert dist == pytest.approx(ref)

    def test_not_collinear(self, host_anchor, universe):
        # check none of the g2-g1-g0-h0-h1-h2 vectors are not collinear
        assert not is_collinear(
            positions=np.vstack((
                universe.atoms[[5528, 5507, 5508]].positions,
                universe.atoms[host_anchor].positions
            )),
            atoms=[0, 1, 2, 3, 4, 5],
            dimensions=universe.dimensions
        )

    def test_angles(self, host_anchor, universe):
        # check that the angles aren't flat
        ag1 = mda.lib.distances.calc_angles(
            universe.atoms[5507].position,
            universe.atoms[5528].position,
            universe.atoms[host_anchor[0]].position,
            box=universe.dimensions
        )
        ag2 = mda.lib.distances.calc_angles(
            universe.atoms[5528].position,
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
            universe.atoms[5508].position,
            universe.atoms[5507].position,
            universe.atoms[5528].position,
            universe.atoms[host_anchor[0]].position,
            box=universe.dimensions,
        )
        dih2 = mda.lib.distances.calc_dihedrals(
            universe.atoms[5507].position,
            universe.atoms[5528].position,
            universe.atoms[host_anchor[0]].position,
            universe.atoms[host_anchor[1]].position,
            box=universe.dimensions,
        )
        dih3 = mda.lib.distances.calc_dihedrals(
            universe.atoms[5528].position,
            universe.atoms[host_anchor[0]].position,
            universe.atoms[host_anchor[1]].position,
            universe.atoms[host_anchor[2]].position,
            box=universe.dimensions,
        )
        assert check_dihedral_bounds(dih1 * unit.radians)
        assert check_dihedral_bounds(dih2 * unit.radians)
        assert check_dihedral_bounds(dih3 * unit.radians)

@pytest.mark.slow
def test_find_host_anchor_none(eg5_protein_ligand_universe):

    host_anchor = find_host_anchor(
        guest_atoms=eg5_protein_ligand_universe.atoms[[5528, 5507, 5508]],
        host_atom_pool=eg5_protein_ligand_universe.select_atoms("backbone"),
        host_minimum_distance=4.5 * unit.nanometer,
        guest_minimum_distance=4.5 * unit.nanometer,
        angle_force_constant=83.68 * unit.kilojoule_per_mole / unit.radians**2,
        temperature=298.15 * unit.kelvin,
    )
    # we should get None if no atoms can be found
    assert host_anchor is None
