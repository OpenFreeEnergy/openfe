# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import itertools
import lomap
import math
import numpy as np
from numpy.testing import assert_allclose
import openfe
from openfe.setup import lomap_scorers
import pytest
from rdkit import Chem
from rdkit.Chem.AllChem import Compute2DCoords


@pytest.fixture()
def toluene_to_cyclohexane(lomap_basic_test_files):
    meth = lomap_basic_test_files['methylcyclohexane']
    tolu = lomap_basic_test_files['toluene']
    mapping = [(0, 0), (1, 1), (2, 6), (3, 5), (4, 4), (5, 3), (6, 2)]

    return openfe.setup.LigandAtomMapping(tolu, meth,
                                          molA_to_molB=dict(mapping))


@pytest.fixture()
def toluene_to_methylnaphthalene(lomap_basic_test_files):
    tolu = lomap_basic_test_files['toluene']
    naph = lomap_basic_test_files['2-methylnaphthalene']
    mapping = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 8), (5, 9), (6, 10)]

    return openfe.setup.LigandAtomMapping(tolu, naph,
                                          molA_to_molB=dict(mapping))


@pytest.fixture()
def toluene_to_heptane(lomap_basic_test_files):
    tolu = lomap_basic_test_files['toluene']
    hept = Chem.MolFromSmiles('CCCCCCC')
    Chem.rdDepictor.Compute2DCoords(hept)
    hept = openfe.setup.SmallMoleculeComponent(hept)

    mapping = [(6, 0)]

    return openfe.setup.LigandAtomMapping(tolu, hept,
                                          molA_to_molB=dict(mapping))


@pytest.fixture()
def methylnaphthalene_to_naphthol(lomap_basic_test_files):
    m1 = lomap_basic_test_files['2-methylnaphthalene']
    m2 = lomap_basic_test_files['2-naftanol']
    mapping = [(0, 0), (1, 1), (2, 10), (3, 9), (4, 8), (5, 7), (6, 6), (7, 5),
               (8, 4), (9, 3), (10, 2)]

    return openfe.setup.LigandAtomMapping(m1, m2, molA_to_molB=dict(mapping))


def test_mcsr_zero(toluene_to_cyclohexane):
    score = lomap_scorers.mcsr_score(toluene_to_cyclohexane)

    # all atoms map, so perfect score
    assert score == 0


def test_mcsr_nonzero(toluene_to_methylnaphthalene):
    score = lomap_scorers.mcsr_score(toluene_to_methylnaphthalene)

    assert score == pytest.approx(1 - math.exp(-0.1 * 4))


def test_mcsr_custom_beta(toluene_to_methylnaphthalene):
    score = lomap_scorers.mcsr_score(toluene_to_methylnaphthalene, beta=0.2)

    assert score == pytest.approx(1 - math.exp(-0.2 * 4))


def test_mcnar_score_pass(toluene_to_cyclohexane):
    score = lomap_scorers.mncar_score(toluene_to_cyclohexane)

    assert score == 0


def test_mcnar_score_fail(toluene_to_heptane):
    score = lomap_scorers.mncar_score(toluene_to_heptane)

    assert score == 1.0


def test_atomic_number_score_pass(toluene_to_cyclohexane):
    score = lomap_scorers.atomic_number_score(toluene_to_cyclohexane)

    assert score == 0.0


def test_atomic_number_score_fail(methylnaphthalene_to_naphthol):
    score = lomap_scorers.atomic_number_score(
        methylnaphthalene_to_naphthol)

    # single mismatch @ 0.5
    assert score == pytest.approx(1 - math.exp(-0.1 * 0.5))


def test_atomic_number_score_weights(methylnaphthalene_to_naphthol):
    difficulty = {
        8: {6: 0.75},  # oxygen to carbon @ 12
    }

    score = lomap_scorers.atomic_number_score(
        methylnaphthalene_to_naphthol, difficulty=difficulty)

    # single mismatch @ (1 - 0.75)
    assert score == pytest.approx(1 - math.exp(-0.1 * 0.25))


class TestSulfonamideRule:
    @staticmethod
    @pytest.fixture
    def ethylbenzene():
        m = Chem.AddHs(Chem.MolFromSmiles('c1ccccc1CCC'))

        return openfe.setup.SmallMoleculeComponent.from_rdkit(m)

    @staticmethod
    @pytest.fixture
    def sulfonamide():
        # technically 3-phenylbutane-1-sulfonamide
        m = Chem.AddHs(Chem.MolFromSmiles('c1ccccc1C(C)CCS(=O)(=O)N'))

        return openfe.setup.SmallMoleculeComponent.from_rdkit(m)

    @staticmethod
    @pytest.fixture
    def from_sulf_mapping():
        # this is the standard output from lomap_scorers
        return {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 14,
                8: 7, 9: 8, 10: 18, 14: 9, 15: 10, 16: 11, 17: 12,
                18: 13, 19: 15, 23: 16, 24: 17, 25: 19, 26: 20}

    @staticmethod
    def test_sulfonamide_hit_backwards(ethylbenzene, sulfonamide,
                                       from_sulf_mapping):
        # a sulfonamide completely disappears on the RHS, so should trigger
        # the sulfonamide score to try and forbid this

        mapping = openfe.setup.LigandAtomMapping(
            molA=sulfonamide,
            molB=ethylbenzene,
            molA_to_molB=from_sulf_mapping,
        )
        expected = 1 - math.exp(-1 * 0.4)
        assert lomap_scorers.sulfonamides_score(mapping) == expected

    @staticmethod
    def test_sulfonamide_hit_forwards(ethylbenzene, sulfonamide,
                                      from_sulf_mapping):
        AtoB = {v: k for k, v in from_sulf_mapping.items()}

        # this is the standard output from lomap_scorers
        mapping = openfe.setup.LigandAtomMapping(molA=ethylbenzene,
                                                 molB=sulfonamide,
                                                 molA_to_molB=AtoB)

        expected = 1 - math.exp(-1 * 0.4)
        assert lomap_scorers.sulfonamides_score(mapping) == expected


@pytest.mark.parametrize('base,other,name,hit', [
    ('CCc1ccccc1', 'CCc1ccc(-c2ccco2)cc1', 'phenylfuran', False),
    ('CCc1ccccc1', 'CCc1ccc(-c2cnc[nH]2)cc1', 'phenylimidazole', True),
    ('CCc1ccccc1', 'CCc1ccc(-c2ccno2)cc1', 'phenylisoxazole', True),
    ('CCc1ccccc1', 'CCc1ccc(-c2cnco2)cc1', 'phenyloxazole', True),
    ('CCc1ccccc1', 'CCc1ccc(-c2cccnc2)cc1', 'phenylpyridine1', True),
    ('CCc1ccccc1', 'CCc1ccc(-c2ccccn2)cc1', 'phenylpyridine2', True),
    ('CCc1ccccc1', 'CCc1ccc(-c2cncnc2)cc1', 'phenylpyrimidine', True),
    ('CCc1ccccc1', 'CCc1ccc(-c2ccc[nH]2)cc1', 'phenylpyrrole', False),
    ('CCc1ccccc1', 'CCc1ccc(-c2ccccc2)cc1', 'phenylphenyl', False),
])
def test_heterocycle_score(base, other, name, hit):
    # base -> other transform, if *hit* a forbidden heterocycle is created
    r1 = Chem.AddHs(Chem.MolFromSmiles(base))
    r2 = Chem.AddHs(Chem.MolFromSmiles(other))
    # add 2d coords to stop Lomap crashing for now
    for r in [r1, r2]:
        Compute2DCoords(r)
    m1 = openfe.setup.SmallMoleculeComponent.from_rdkit(r1)
    m2 = openfe.setup.SmallMoleculeComponent.from_rdkit(r2)

    mapper = openfe.setup.LomapAtomMapper(threed=False)
    mapping = next(mapper.suggest_mappings(m1, m2))
    score = lomap_scorers.heterocycles_score(mapping)

    assert score == 0 if not hit else score == 1 - math.exp(-0.4)


# test individual scoring functions against lomap
SCORE_NAMES = {
    'mcsr': 'mcsr_score',
    'mncar': 'mncar_score',
    'atomic_number_rule': 'atomic_number_score',
    'hybridization_rule': 'hybridization_score',
    'sulfonamides_rule': 'sulfonamides_score',
    'heterocycles_rule': 'heterocycles_score',
    'transmuting_methyl_into_ring_rule': 'transmuting_methyl_into_ring_score',
    'transmuting_ring_sizes_rule': 'transmuting_ring_sizes_score'
}
IX = itertools.combinations(range(8), 2)


@pytest.mark.parametrize('params', itertools.product(SCORE_NAMES, IX))
def test_lomap_individual_scores(params,
                                 lomap_basic_test_files):
    scorename, (i, j) = params
    mols = sorted(lomap_basic_test_files.items())
    _, molA = mols[i]
    _, molB = mols[j]

    # reference value
    lomap_version = getattr(lomap.MCS(molA.to_rdkit(),
                                      molB.to_rdkit()), scorename)()

    # longer way
    mapper = openfe.setup.LomapAtomMapper(threed=False)
    mapping = next(mapper.suggest_mappings(molA, molB))
    openfe_version = getattr(lomap_scorers, SCORE_NAMES[scorename])(mapping)

    assert lomap_version == pytest.approx(1 - openfe_version), \
           f"{molA.name} {molB.name} {scorename}"


# full back to back test again lomap
def test_lomap_regression(lomap_basic_test_files_dir,  # in a dir for lomap
                          lomap_basic_test_files):
    # run lomap
    dbmols = lomap.DBMolecules(lomap_basic_test_files_dir)
    matrix, _ = dbmols.build_matrices()
    matrix = matrix.to_numpy_2D_array()

    assert matrix.shape == (8, 8)

    # now run the openfe equivalent
    # first, get the order identical to lomap
    smallmols = []
    for i in range(matrix.shape[0]):
        nm = dbmols[i].getName()
        smallmols.append(lomap_basic_test_files[nm[:-5]])  # - ".mol2"

    mapper = openfe.setup.LomapAtomMapper(threed=False)
    scorer = lomap_scorers.default_lomap_score
    scores = np.zeros_like(matrix)
    for i, j in itertools.combinations(range(matrix.shape[0]), 2):
        molA = smallmols[i]
        molB = smallmols[j]

        mapping = next(mapper.suggest_mappings(molA, molB))
        score = scorer(mapping)

        scores[i, j] = scores[j, i] = score
    scores = 1 - scores
    # fudge diagonal for comparison
    for i in range(matrix.shape[0]):
        scores[i, i] = 0

    assert_allclose(matrix, scores)
