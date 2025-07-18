# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import itertools
import os
import pathlib

import MDAnalysis as mda
import numpy as np
import pooch
import pytest
from openfe.protocols.restraint_utils.geometry.utils import (
    CentroidDistanceSort,
    FindHostAtoms,
    _atomgroup_has_bonds,
    _get_mda_selection,
    _wrap_angle,
    check_angle_not_flat,
    check_angular_variance,
    check_dihedral_bounds,
    get_aromatic_atom_idxs,
    get_aromatic_rings,
    get_central_atom_idx,
    get_heavy_atom_idxs,
    get_local_rmsf,
    is_collinear,
    protein_chain_selection,
    stable_secondary_structure_selection,
)
from openff.units import unit
from rdkit import Chem

from ...conftest import HAS_INTERNET


@pytest.fixture(scope="module")
def eg5_pdb_universe(eg5_protein_pdb):
    return mda.Universe(eg5_protein_pdb)


@pytest.fixture
def eg5_protein_ligand_universe(eg5_protein_pdb, eg5_ligands):
    protein = mda.Universe(eg5_protein_pdb)
    lig = mda.Universe(eg5_ligands[1].to_rdkit())
    # add the residue name of the ligand
    lig.add_TopologyAttr("resname", ["LIG"])
    return mda.Merge(protein.atoms, lig.atoms)


POOCH_CACHE = pooch.os_cache("openfe")
zenodo_restraint_data = pooch.create(
    path=POOCH_CACHE,
    base_url="doi:10.5281/zenodo.15212342",
    registry={
        "t4_lysozyme_trajectory.zip": "sha256:e985d055db25b5468491e169948f641833a5fbb67a23dbb0a00b57fb7c0e59c8"
    },
    retry_if_failed=3,
)


@pytest.fixture
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
    return universe


def test_mda_selection_none_error(eg5_pdb_universe):
    with pytest.raises(ValueError, match="one of either"):
        _ = _get_mda_selection(eg5_pdb_universe)


def test_mda_selection_both_args_error(eg5_pdb_universe):
    with pytest.raises(ValueError, match="both atom_list and"):
        _ = _get_mda_selection(
            eg5_pdb_universe, atom_list=[0, 1, 2, 3], selection="all"
        )


def test_mda_selection_universe_atom_list(eg5_pdb_universe):
    test_ag = _get_mda_selection(eg5_pdb_universe, atom_list=[0, 1, 2])
    assert eg5_pdb_universe.atoms[[0, 1, 2]] == test_ag


def test_mda_selection_atomgroup_string(eg5_pdb_universe):
    # test that the selection is reducing the atom group
    test_ag = _get_mda_selection(eg5_pdb_universe.atoms, selection="protein")
    assert test_ag != eg5_pdb_universe.atoms
    assert test_ag.n_atoms == 5474


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ["C1CCCCC1", []],
        ["[C@@H]1([C@@H]([C@@H](OC([C@@H]1O)O)C(=O)O)O)O", []],
        ["C1=CC=CC=C1", [6]],
        ["C1=CC2C=CC1C=C2", [8]],
        ["C1CC2=CC=CC=C2C1", [6]],
        ["C1=COC=C1", [5]],
        ["C1=CC=C2C=CC=CC2=C1", [10]],
        ["C1=CC=C(C=C1)C2=CC=CC=C2", [6, 6]],
        ["C1=CC=C(C=C1)C(C2=CC=CC=C2)(C3=CC=CC=C3Cl)N4C=CN=C4", [6, 6, 6, 5]],
    ],
)
def test_aromatic_rings(smiles, expected):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

    # get the rings
    rings = get_aromatic_rings(mol)

    # check we have the right number of rings & their size
    for i, r in enumerate(rings):
        assert len(r) == expected[i]

    # check that there is no overlap in atoms between each ring
    for x, y in itertools.combinations(rings, 2):
        assert x.isdisjoint(y)

    # get the aromatic idx
    arom_idxs = get_aromatic_atom_idxs(mol)

    # Check that all the ring indices are aromatic
    assert all(idx in arom_idxs for idx in itertools.chain(*rings))

    # Also check the lengths match
    assert sum(len(r) for r in rings) == len(arom_idxs)

    # Finally check that all the arom_idxs are actually aromatic
    for idx in arom_idxs:
        at = mol.GetAtomWithIdx(idx)
        assert at.GetIsAromatic()


@pytest.mark.parametrize(
    "smiles, nheavy, nlight",
    [
        ["C1CCCCC1", 6, 12],
        ["[C@@H]1([C@@H]([C@@H](OC([C@@H]1O)O)C(=O)O)O)O", 13, 10],
        ["C1=CC=CC=C1", 6, 6],
        ["C1=CC2C=CC1C=C2", 8, 8],
        ["C1CC2=CC=CC=C2C1", 9, 10],
        ["C1=COC=C1", 5, 4],
        ["C1=CC=C2C=CC=CC2=C1", 10, 8],
        ["C1=CC=C(C=C1)C2=CC=CC=C2", 12, 10],
        ["C1=CC=C(C=C1)C(C2=CC=CC=C2)(C3=CC=CC=C3Cl)N4C=CN=C4", 25, 17],
    ],
)
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


@pytest.mark.parametrize(
    "smiles, idx",
    [
        ["C1CCCCC1", 2],
        ["[C@@H]1([C@@H]([C@@H](OC([C@@H]1O)O)C(=O)O)O)O", 3],
        ["C1=CC=CC=C1", 2],
        ["C1=CC2C=CC1C=C2", 2],
        ["C1CC2=CC=CC=C2C1", 2],
        ["C1=COC=C1", 4],
        ["C1=CC=C2C=CC=CC2=C1", 3],
        ["C1=CC=C(C=C1)C2=CC=CC=C2", 6],
        ["C1=CC=C(C=C1)C(C2=CC=CC=C2)(C3=CC=CC=C3Cl)N4C=CN=C4", 6],
        ["OC(COc1ccc(cc1)CC(=O)N)CNC(C)C", 3],
    ],
)
def test_central_idx(smiles, idx):
    """
    Regression tests for getting central atom idx.
    """
    rdmol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert get_central_atom_idx(rdmol) == idx


def test_central_atom_disconnected():
    mol = Chem.AddHs(Chem.MolFromSmiles("C.C"))

    with pytest.raises(ValueError, match="disconnected molecule"):
        _ = get_central_atom_idx(mol)


def test_collinear_too_few_atoms():
    with pytest.raises(ValueError, match="Too few atoms passed"):
        _ = is_collinear(None, [1, 2], None)


def test_collinear_index_match_error_length():
    with pytest.raises(ValueError, match="indices do not match"):
        _ = is_collinear(
            positions=np.zeros((3, 3)),
            atoms=[0, 1, 2, 3],
        )


def test_collinear_index_match_error_index():
    with pytest.raises(ValueError, match="atoms is not a list of index integers"):
        _ = is_collinear(
            positions=np.zeros((4, 3)),
            atoms=[1, 2.5, 3],
        )


@pytest.mark.parametrize(
    "arr, thresh, truth",
    [
        [[[0, 0, -1], [1, 0, 0], [2, 0, 1]], 0.9, True],
        [[[0, 0, -1], [1, 0, 0], [2, 0, 2]], 0.9, True],
        [[[0, 0, -1], [1, 0, 0], [2, 0, 2]], 0.95, False],
        [[[0, 1, -1], [1, 0, 0], [2, 0, 1]], 0.9, False],
        [[[0, 1, -1], [1, 1, 0], [2, 1, 1]], 0.9, True],
        [[[0, 1, -1], [1, 1, 0], [2, 1, 2]], 0.95, False],
        [[[0, 0, -1], [1, 1, 0], [2, 2, 1]], 0.95, True],
        [[[0, 0, -1], [1, 0, 0], [2, 0, 1]], 0.95, True],
        [[[2, 0, -1], [1, 0, 0], [0, 0, 1]], 0.95, True],
        [[[0, 0, 1], [0, 0, 0], [0, 0, 2]], 0.95, True],
        [[[1, 1, 1], [0, 0, 0], [2, 2, 2]], 0.9, True],
    ],
)
def test_is_collinear_three_atoms(arr, thresh, truth):
    assert (
        is_collinear(positions=np.array(arr), atoms=[0, 1, 2], threshold=thresh)
        == truth
    )


@pytest.mark.parametrize(
    "arr, truth, dims",
    [
        [[[0, 0, -1], [1, 0, 0], [2, 0, 1]], True, True],
        [[[0, 0, -1], [1, 0, 0], [2, 0, 11]], False, False],
        [[[0, 0, -1], [1, 0, 0], [2, 0, 11]], True, True],
        [[[0, 0, -1], [1, 0, 0], [2, 0, 2]], False, True],
        [[[0, 0, -1], [1, 0, 0], [2, 0, 12]], False, False],
        [[[0, 0, -1], [1, 0, 0], [2, 0, 12]], False, True],
    ],
)
def test_is_collinear_three_atoms_dimensions(arr, truth, dims):
    if dims:
        dimensions = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0], dtype=float)
    else:
        dimensions = None

    assert (
        is_collinear(
            positions=np.array(arr, dtype=float),
            atoms=[0, 1, 2],
            threshold=0.99,
            dimensions=dimensions,
        )
        == truth
    )


@pytest.mark.parametrize(
    "arr, tresh, truth",
    [
        # collinear all
        [[[0, 0, -1], [1, 0, 0], [2, 0, 1], [3, 0, 2]], 0.99, True],
        # not collinear for all
        [[[0, 0, -1], [1, 0, 0], [2, 0, 2], [3, 0, 2]], 0.99, False],
        # not collinear but within threshold
        [[[0, 0, -1], [1, 0, 0], [2, 0, 2], [3, 0, 2]], 0.9, True],
        # collinear for v3 and v4
        [[[0, 0, -1], [1, 0, 0], [2, 0, 2], [3, 0, 4]], 0.99, True],
        # collinear for v1 and v2
        [[[0, 0, -1], [1, 0, 0], [2, 0, 1], [3, 0, 4]], 0.99, True],
        # not collinear for all
        [[[0, 1, -1], [1, 0, 0], [2, 0, 2], [3, 0, 2]], 0.99, False],
    ],
)
def test_is_collinear_four_atoms(arr, tresh, truth):
    assert (
        is_collinear(
            positions=np.array(arr),
            atoms=[0, 1, 2, 3],
            threshold=tresh,
        )
        == truth
    )


def test_wrap_angle_degrees():
    for i in range(0, 361, 1):
        angle = _wrap_angle(i * unit.degrees)
        if i > 180:
            expected = ((i - 360) * unit.degrees).to("radians").m
        else:
            expected = (i * unit.degrees).to("radians").m

        assert angle.m == pytest.approx(expected)


@pytest.mark.parametrize(
    "angle, expected",
    [
        [0 * unit.radians, 0 * unit.radians],
        [1 * unit.radians, 1 * unit.radians],
        [4 * unit.radians, 4 - (2 * np.pi) * unit.radians],
        [-4 * unit.radians, -4 + (2 * np.pi) * unit.radians],
    ],
)
def test_wrap_angle_radians(angle, expected):
    assert _wrap_angle(angle) == pytest.approx(expected)


@pytest.mark.parametrize(
    "limit, force, temperature",
    [
        [0.7695366605411506, 83.68, 298.15],
        [0.8339791717799163, 83.68, 350.0],
        [0.5441445910402979, 167.36, 298.15],
    ],
)
def test_angle_not_flat(limit, force, temperature):
    limit = limit * unit.radians
    force = force * unit.kilojoule_per_mole / unit.radians**2
    temperature = temperature * unit.kelvin

    # test upper
    assert check_angle_not_flat(limit + 0.01, force, temperature)
    assert not check_angle_not_flat(limit - 0.01, force, temperature)

    # test lower
    limit = np.pi - limit
    assert check_angle_not_flat(limit - 0.01, force, temperature)
    assert not check_angle_not_flat(limit + 0.01, force, temperature)


@pytest.mark.parametrize(
    "dihed, expected",
    [
        [3 * unit.radians, False],
        [0 * unit.radians, True],
        [-3 * unit.radians, False],
        [300 * unit.degrees, True],
        [181 * unit.degrees, False],
    ],
)
def test_check_dihedral_bounds(dihed, expected):
    ret = check_dihedral_bounds(dihed)
    assert ret == expected


@pytest.mark.parametrize(
    "dihed, lower, upper, expected",
    [
        [3 * unit.radians, -3.1 * unit.radians, 3.1 * unit.radians, True],
        [300 * unit.degrees, -61 * unit.degrees, 301 * unit.degrees, True],
        [300 * unit.degrees, 299 * unit.degrees, -61 * unit.degrees, False],
    ],
)
def test_check_dihedral_bounds_defined(dihed, lower, upper, expected):
    ret = check_dihedral_bounds(dihed, lower_cutoff=lower, upper_cutoff=upper)
    assert ret == expected


def test_angular_variance():
    """
    Manual check with for an input number of angles with
    a known variance of 0.36216
    """
    angles = [0, 1, 2, 6]

    assert check_angular_variance(
        angles=angles * unit.radians,
        upper_bound=np.pi * unit.radians,
        lower_bound=-np.pi * unit.radians,
        width=0.37 * unit.radians,
    )

    assert not check_angular_variance(
        angles=angles * unit.radians,
        upper_bound=np.pi * unit.radians,
        lower_bound=-np.pi * unit.radians,
        width=0.35 * unit.radians,
    )


def test_atomgroup_has_bonds(eg5_protein_pdb):
    # Creating a new universe because we'll modify this one
    u = mda.Universe(eg5_protein_pdb)

    # PDB has water bonds
    assert len(u.bonds) == 14
    assert _atomgroup_has_bonds(u) is False
    assert _atomgroup_has_bonds(u.select_atoms("resname HOH")) is True

    # Delete the topology attr and everything is false
    u.del_TopologyAttr("bonds")
    assert _atomgroup_has_bonds(u) is False
    assert _atomgroup_has_bonds(u.select_atoms("resname HOH")) is False

    # Guess some bonds back
    ag = u.atoms[:100]
    ag.guess_bonds()
    assert _atomgroup_has_bonds(ag) is True


def test_centroid_distance_sort(eg5_protein_ligand_universe):

    # quickly sort the atoms of the first residue
    atom_sort = CentroidDistanceSort(
        sortable_atoms=eg5_protein_ligand_universe.select_atoms(
            "backbone and resnum 15"
        ),
        reference_atoms=eg5_protein_ligand_universe.select_atoms("resname LIG"),
    )
    atom_sort.run()
    sorted_ids = [a.ix for a in atom_sort.results.sorted_atomgroup]
    # hard code the ids we expect
    assert sorted_ids == [2, 1]


def test_find_host_atoms(eg5_protein_ligand_universe):

    # very small window to limit atoms for speed
    min_cutoff = 1 * unit.nanometer
    max_cutoff = 1.1 * unit.nanometer

    atom_finder = FindHostAtoms(
        host_atoms=eg5_protein_ligand_universe.select_atoms("backbone"),
        # hand picked ring atom
        guest_atoms=eg5_protein_ligand_universe.atoms[5528],
        min_search_distance=min_cutoff,
        max_search_distance=max_cutoff,
    )
    atom_finder.min_cutoff == min_cutoff.to("angstrom").m
    atom_finder.max_cutoff == max_cutoff.to("angstrom").m

    atom_finder.run()
    # should find the 28 close backbone atoms
    assert len(atom_finder.results.host_idxs) == 28


def test_get_rmsf_single_frame(eg5_protein_ligand_universe):
    ligand = eg5_protein_ligand_universe.select_atoms("resname LIG")
    rmsf = get_local_rmsf(atomgroup=ligand)
    # as we have a single frame we should get all zeros back
    assert np.allclose(rmsf.m, np.zeros(ligand.n_atoms))


@pytest.mark.skipif(
    not os.path.exists(POOCH_CACHE) and not HAS_INTERNET,
    reason="Internet seems to be unavailable and test data is not cached locally.",
)
def test_get_rmsf_trajectory(t4_lysozyme_trajectory_universe):
    # get the RMSF of just the ligand
    ligand = t4_lysozyme_trajectory_universe.select_atoms("resname UNK")
    rmsf = get_local_rmsf(atomgroup=ligand)
    assert len(rmsf) == ligand.n_atoms
    # regression test the calculation of the rmsf
    assert np.allclose(
        rmsf.to("angstrom").m,
        np.array(
            [
                0.054697843819888965,
                0.07512308066036011,
                0.06000502046267635,
                0.07180001811557828,
                0.043416981393784,
                0.05909972948285153,
                0.13061051498104648,
                0.15166255437235665,
                0.17860692733595412,
                0.1483198866730507,
                0.14193714526412668,
                0.06730488032625732,
                1.0235330857263523,
                1.0048466548200004,
                1.0209553834502236,
            ]
        ),
    )


@pytest.mark.skipif(
    not os.path.exists(POOCH_CACHE) and not HAS_INTERNET,
    reason="Internet seems to be unavailable and test data is not cached locally.",
)
def test_stable_ss_selection(t4_lysozyme_trajectory_universe):

    # use a copy as we need to remove the bonds first
    universe_copy = t4_lysozyme_trajectory_universe.copy()
    universe_copy.del_TopologyAttr("bonds")

    ligand = universe_copy.select_atoms("resname LIG")

    with pytest.warns(
        match="No bonds found in input Universe, will attempt to guess them."
    ):
        stable_protein = stable_secondary_structure_selection(
            # DDSP should filter by protein we will check at the end
            atomgroup=universe_copy.atoms,
        )
        # make sure the ligand is not in this selection
        overlapping_ligand = stable_protein.intersection(ligand.atoms)
        assert overlapping_ligand.n_atoms == 0
        # make sure we get the expected number of atoms
        assert stable_protein.n_atoms == 780


def test_protein_chain_selection(eg5_protein_ligand_universe):

    # use a copy as we need to remove the bonds first
    universe_copy = eg5_protein_ligand_universe.copy()
    universe_copy.del_TopologyAttr("bonds")

    ligand = universe_copy.select_atoms("resname LIG")

    with pytest.warns(
        match="No bonds found in input Universe, will attempt to guess them."
    ):
        chain_selection = protein_chain_selection(
            # the selection should filter for the protein we will check at the end
            atomgroup=universe_copy.atoms,
        )
        overlapping_ligand = chain_selection.intersection(ligand.atoms)
        assert overlapping_ligand.n_atoms == 0
        # make sure we get the expected number of atoms
        assert chain_selection.n_atoms == 5150


def test_protein_chain_selection_subchain(eg5_pdb_universe):
    """
    Pass a subset of the residues in a chain
    """

    sele = protein_chain_selection(
        atomgroup=eg5_pdb_universe.residues[:28].atoms,
    )

    assert len(sele.residues) == 18
    assert len(sele) == 282


def test_protein_chain_selection_nochains(eg5_pdb_universe):
    """
    Artificially bump up the minimum number of residues per
    chain such that we don't have any chains.
    """

    sele = protein_chain_selection(
        atomgroup=eg5_pdb_universe.atoms,
        min_chain_length=99999,
    )

    assert len(sele) == 0

def test_protein_chain_selection_trim_too_large(eg5_pdb_universe):
    """
    Use artificially large trim sizes that are greater than the length of the residue.
    """

    sele = protein_chain_selection(
        atomgroup=eg5_pdb_universe.atoms,
        min_chain_length=30,
        trim_chain_start=5000,
        trim_chain_end=8000,
    )

    assert len(sele) == 0
