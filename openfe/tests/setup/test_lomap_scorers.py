# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import math
import openfe
import pytest
from rdkit import Chem


@pytest.fixture()
def toluene_to_cyclohexane(lomap_basic_test_files):
    meth = lomap_basic_test_files['methylcyclohexane']
    tolu = lomap_basic_test_files['toluene']
    mapping = [(0, 0), (1, 1), (2, 6), (3, 5), (4, 4), (5, 3), (6, 2)]

    return openfe.setup.AtomMapping(tolu, meth, mol1_to_mol2=dict(mapping))


@pytest.fixture()
def toluene_to_methylnaphthalene(lomap_basic_test_files):
    tolu = lomap_basic_test_files['toluene']
    naph = lomap_basic_test_files['2-methylnaphthalene']
    mapping = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 8), (5, 9), (6, 10)]

    return openfe.setup.AtomMapping(tolu, naph, mol1_to_mol2=dict(mapping))


@pytest.fixture()
def toluene_to_heptane(lomap_basic_test_files):
    tolu = lomap_basic_test_files['toluene']
    hept = Chem.MolFromSmiles('CCCCCCC')
    Chem.rdDepictor.Compute2DCoords(hept)
    hept = openfe.setup.Molecule(hept)

    mapping = [(6, 0)]

    return openfe.setup.AtomMapping(tolu, hept, mol1_to_mol2=dict(mapping))


class TestScorer:
    def test_mcsr_zero(self, toluene_to_cyclohexane):
        score = openfe.setup.LomapAtomMapper.mcsr_score(toluene_to_cyclohexane)

        # all atoms map, so perfect score
        assert score == 0

    def test_mcsr_nonzero(self, toluene_to_methylnaphthalene):
        score = openfe.setup.LomapAtomMapper.mcsr_score(
            toluene_to_methylnaphthalene)

        assert score == pytest.approx(1 - math.exp(-0.1 * 4))

    def test_mcsr_custom_beta(self, toluene_to_methylnaphthalene):
        score = openfe.setup.LomapAtomMapper.mcsr_score(
            toluene_to_methylnaphthalene, beta=0.2)

        assert score == pytest.approx(1 - math.exp(-0.2 * 4))

    def test_mcnar_score_pass(self, toluene_to_cyclohexane):
        score = openfe.setup.LomapAtomMapper.mcnar_score(toluene_to_cyclohexane)

        assert score == 0

    def test_mcnar_score_fail(self, toluene_to_heptane):
        score = openfe.setup.LomapAtomMapper.mcnar_score(toluene_to_heptane)

        assert score == float('inf')
