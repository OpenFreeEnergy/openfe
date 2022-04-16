# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import math
import openfe
from openfe.setup import LomapAtomMapper
import pytest
from rdkit import Chem


@pytest.fixture()
def toluene_to_cyclohexane(lomap_basic_test_files):
    meth = lomap_basic_test_files['methylcyclohexane']
    tolu = lomap_basic_test_files['toluene']
    mapping = [(0, 0), (1, 1), (2, 6), (3, 5), (4, 4), (5, 3), (6, 2)]

    return openfe.setup.LigandAtomMapping(tolu, meth, molA_to_molB=dict(mapping))


@pytest.fixture()
def toluene_to_methylnaphthalene(lomap_basic_test_files):
    tolu = lomap_basic_test_files['toluene']
    naph = lomap_basic_test_files['2-methylnaphthalene']
    mapping = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 8), (5, 9), (6, 10)]

    return openfe.setup.LigandAtomMapping(tolu, naph, molA_to_molB=dict(mapping))


@pytest.fixture()
def toluene_to_heptane(lomap_basic_test_files):
    tolu = lomap_basic_test_files['toluene']
    hept = Chem.MolFromSmiles('CCCCCCC')
    Chem.rdDepictor.Compute2DCoords(hept)
    hept = openfe.setup.SmallMoleculeComponent(hept)

    mapping = [(6, 0)]

    return openfe.setup.LigandAtomMapping(tolu, hept, molA_to_molB=dict(mapping))


@pytest.fixture()
def methylnaphthalene_to_naphthol(lomap_basic_test_files):
    m1 = lomap_basic_test_files['2-methylnaphthalene']
    m2 = lomap_basic_test_files['2-naftanol']
    mapping = [(0, 0), (1, 1), (2, 10), (3, 9), (4, 8), (5, 7), (6, 6), (7, 5),
               (8, 4), (9, 3), (10, 2)]

    return openfe.setup.LigandAtomMapping(m1, m2, molA_to_molB=dict(mapping))


class TestScorer:
    def test_mcsr_zero(self, toluene_to_cyclohexane):
        score = LomapAtomMapper.mcsr_score(toluene_to_cyclohexane)

        # all atoms map, so perfect score
        assert score == 0

    def test_mcsr_nonzero(self, toluene_to_methylnaphthalene):
        score = LomapAtomMapper.mcsr_score(toluene_to_methylnaphthalene)

        assert score == pytest.approx(1 - math.exp(-0.1 * 4))

    def test_mcsr_custom_beta(self, toluene_to_methylnaphthalene):
        score = LomapAtomMapper.mcsr_score(toluene_to_methylnaphthalene,
                                           beta=0.2)

        assert score == pytest.approx(1 - math.exp(-0.2 * 4))

    def test_mcnar_score_pass(self, toluene_to_cyclohexane):
        score = LomapAtomMapper.mcnar_score(toluene_to_cyclohexane)

        assert score == 0

    def test_mcnar_score_fail(self, toluene_to_heptane):
        score = LomapAtomMapper.mcnar_score(toluene_to_heptane)

        assert score == float('inf')

    def test_atomic_number_score_pass(self, toluene_to_cyclohexane):
        score = LomapAtomMapper.atomic_number_score(toluene_to_cyclohexane)

        assert score == 0.0

    def test_atomic_number_score_fail(self, methylnaphthalene_to_naphthol):
        score = LomapAtomMapper.atomic_number_score(
            methylnaphthalene_to_naphthol)

        # single mismatch @ 0.5
        assert score == pytest.approx(1 - math.exp(-0.1 * 0.5))

    def test_atomic_number_score_weights(self, methylnaphthalene_to_naphthol):
        difficulty = {
            8: {6: 0.75},  # oxygen to carbon @ 12
        }

        score = LomapAtomMapper.atomic_number_score(
            methylnaphthalene_to_naphthol, difficulty=difficulty)

        # single mismatch @ (1 - 0.75)
        assert score == pytest.approx(1 - math.exp(-0.1 * 0.25))
