# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import itertools
import lomap
import math
import numpy as np
from numpy.testing import assert_allclose
import openfe
from openfe.setup import lomap_scorers, LigandAtomMapping

import pytest
from rdkit import Chem
from rdkit.Chem.AllChem import Compute2DCoords

from .conftest import mol_from_smiles


@pytest.fixture()
def toluene_to_cyclohexane(atom_mapping_basic_test_files):
    meth = atom_mapping_basic_test_files['methylcyclohexane']
    tolu = atom_mapping_basic_test_files['toluene']
    mapping = [(0, 0), (1, 1), (2, 6), (3, 5), (4, 4), (5, 3), (6, 2)]

    return LigandAtomMapping(tolu, meth,
                             componentA_to_componentB=dict(mapping))


@pytest.fixture()
def toluene_to_methylnaphthalene(atom_mapping_basic_test_files):
    tolu = atom_mapping_basic_test_files['toluene']
    naph = atom_mapping_basic_test_files['2-methylnaphthalene']
    mapping = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 8), (5, 9), (6, 10)]

    return LigandAtomMapping(tolu, naph,
                             componentA_to_componentB=dict(mapping))


@pytest.fixture()
def toluene_to_heptane(atom_mapping_basic_test_files):
    tolu = atom_mapping_basic_test_files['toluene']
    hept = Chem.MolFromSmiles('CCCCCCC')
    Chem.rdDepictor.Compute2DCoords(hept)
    hept = openfe.SmallMoleculeComponent(hept)

    mapping = [(6, 0)]

    return LigandAtomMapping(tolu, hept,
                             componentA_to_componentB=dict(mapping))


@pytest.fixture()
def methylnaphthalene_to_naphthol(atom_mapping_basic_test_files):
    m1 = atom_mapping_basic_test_files['2-methylnaphthalene']
    m2 = atom_mapping_basic_test_files['2-naftanol']
    mapping = [(0, 0), (1, 1), (2, 10), (3, 9), (4, 8), (5, 7), (6, 6), (7, 5),
               (8, 4), (9, 3), (10, 2)]

    return LigandAtomMapping(m1, m2, componentA_to_componentB=dict(mapping))


def test_mcsr_zero(toluene_to_cyclohexane):
    score = lomap_scorers.mcsr_score(toluene_to_cyclohexane)

    # all atoms map, so perfect score
    assert score == 1


def test_mcsr_nonzero(toluene_to_methylnaphthalene):
    score = lomap_scorers.mcsr_score(toluene_to_methylnaphthalene)

    assert score == pytest.approx(math.exp(-0.1 * 4))


def test_mcsr_custom_beta(toluene_to_methylnaphthalene):
    score = lomap_scorers.mcsr_score(toluene_to_methylnaphthalene, beta=0.2)

    assert score == pytest.approx(math.exp(-0.2 * 4))


def test_mcnar_score_pass(toluene_to_cyclohexane):
    score = lomap_scorers.mncar_score(toluene_to_cyclohexane)

    assert score == 1.0


def test_mcnar_score_fail(toluene_to_heptane):
    score = lomap_scorers.mncar_score(toluene_to_heptane)

    assert score == 0.0


def test_atomic_number_score_pass(toluene_to_cyclohexane):
    score = lomap_scorers.atomic_number_score(toluene_to_cyclohexane)

    assert score == 1.0


def test_atomic_number_score_fail(methylnaphthalene_to_naphthol):
    score = lomap_scorers.atomic_number_score(
        methylnaphthalene_to_naphthol)

    # single mismatch @ 0.5
    assert score == pytest.approx(math.exp(-0.1 * 0.5))


def test_atomic_number_score_weights(methylnaphthalene_to_naphthol):
    difficulty = {
        8: {6: 0.75},  # oxygen to carbon @ 12
    }

    score = lomap_scorers.atomic_number_score(
        methylnaphthalene_to_naphthol, difficulty=difficulty)

    # single mismatch @ (1 - 0.75)
    assert score == pytest.approx(math.exp(-0.1 * 0.25))


class TestSulfonamideRule:
    @staticmethod
    @pytest.fixture
    def ethylbenzene():
        m = Chem.AddHs(mol_from_smiles('c1ccccc1CCC'))

        return openfe.SmallMoleculeComponent.from_rdkit(m)

    @staticmethod
    @pytest.fixture
    def sulfonamide():
        # technically 3-phenylbutane-1-sulfonamide
        m = Chem.AddHs(mol_from_smiles('c1ccccc1C(C)CCS(=O)(=O)N'))

        return openfe.SmallMoleculeComponent.from_rdkit(m)

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

        mapping = LigandAtomMapping(
            componentA=sulfonamide,
            componentB=ethylbenzene,
            componentA_to_componentB=from_sulf_mapping,
        )
        expected = math.exp(-1 * 0.4)
        assert lomap_scorers.sulfonamides_score(mapping) == expected

    @staticmethod
    def test_sulfonamide_hit_forwards(ethylbenzene, sulfonamide,
                                      from_sulf_mapping):
        AtoB = {v: k for k, v in from_sulf_mapping.items()}

        # this is the standard output from lomap_scorers
        mapping = LigandAtomMapping(componentA=ethylbenzene,
                                    componentB=sulfonamide,
                                    componentA_to_componentB=AtoB)

        expected = math.exp(-1 * 0.4)
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
    r1 = Chem.AddHs(mol_from_smiles(base))
    r2 = Chem.AddHs(mol_from_smiles(other))
    # add 2d coords to stop Lomap crashing for now
    for r in [r1, r2]:
        Compute2DCoords(r)
    m1 = openfe.SmallMoleculeComponent.from_rdkit(r1)
    m2 = openfe.SmallMoleculeComponent.from_rdkit(r2)

    mapper = openfe.setup.atom_mapping.LomapAtomMapper(
        time=20, threed=False, max3d=1000.0,
        element_change=True, seed='', shift=True,
    )
    mapping = next(mapper.suggest_mappings(m1, m2))
    score = lomap_scorers.heterocycles_score(mapping)

    assert score == 1.0 if not hit else score == math.exp(-0.4)


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
                                 atom_mapping_basic_test_files):
    scorename, (i, j) = params
    mols = sorted(atom_mapping_basic_test_files.items())
    _, molA = mols[i]
    _, molB = mols[j]

    # reference value
    lomap_version = getattr(lomap.MCS(molA.to_rdkit(),
                                      molB.to_rdkit()), scorename)()

    # longer way
    mapper = openfe.setup.atom_mapping.LomapAtomMapper(
        time=20, threed=False, max3d=1000.0,
        element_change=True, seed='', shift=True,
    )
    mapping = next(mapper.suggest_mappings(molA, molB))
    openfe_version = getattr(lomap_scorers, SCORE_NAMES[scorename])(mapping)

    assert lomap_version == pytest.approx(openfe_version), \
           f"{molA.name} {molB.name} {scorename}"


# full back to back test again lomap
def test_lomap_regression(lomap_basic_test_files_dir,  # in a dir for lomap
                          atom_mapping_basic_test_files):
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
        smallmols.append(atom_mapping_basic_test_files[nm[:-5]])  # - ".mol2"

    mapper = openfe.setup.atom_mapping.LomapAtomMapper(
        time=20, threed=False, max3d=1000.0,
        element_change=True, seed='', shift=True,
    )
    scorer = lomap_scorers.default_lomap_score
    scores = np.zeros_like(matrix)
    for i, j in itertools.combinations(range(matrix.shape[0]), 2):
        molA = smallmols[i]
        molB = smallmols[j]

        mapping = next(mapper.suggest_mappings(molA, molB))
        score = scorer(mapping)

        scores[i, j] = scores[j, i] = score
    # fudge diagonal for comparison
    for i in range(matrix.shape[0]):
        scores[i, i] = 0

    assert_allclose(matrix, scores)


def test_transmuting_methyl_into_ring_score():
    """
    Sets up two mappings:
      RC_to_RPh = [CCC]C -> [CCC]Ph
      RH_to_RPh = [CCC]H -> [CCC]Ph
    Where square brackets show mapped (core) region

    The first mapping should trigger this rule, the second shouldn't
    """
    def makemol(smi):
        m = Chem.MolFromSmiles(smi)
        m = Chem.AddHs(m)
        m.Compute2DCoords()

        return openfe.SmallMoleculeComponent(m)

    core = 'CCC{}'
    RC = makemol(core.format('C'))
    RPh = makemol(core.format('c1ccccc1'))
    RH = makemol(core.format('[H]'))

    RC_to_RPh = openfe.LigandAtomMapping(RC, RPh, {i: i for i in range(3)})
    RH_to_RPh = openfe.LigandAtomMapping(RH, RPh, {i: i for i in range(3)})

    score1 = lomap_scorers.transmuting_methyl_into_ring_score(RC_to_RPh)
    score2 = lomap_scorers.transmuting_methyl_into_ring_score(RH_to_RPh)

    assert score1 == pytest.approx(math.exp(-0.1 * 6.0))
    assert score2 == 1.0
