from rdkit import Chem
import pytest

from openfe.setup import AtomMapping


@pytest.fixture(scope='module')
def simple_mapping():
    """Disappearing oxygen on end

    C C O

    C C
    """
    molA = Chem.MolFromSmiles('CCO')
    molB = Chem.MolFromSmiles('CC')

    m = AtomMapping(molA, molB)
    m.AtoB[0] = 0
    m.AtoB[1] = 1
    m.AtoB[2] = None

    m.BtoA[0] = 0
    m.BtoA[1] = 1

    return m


@pytest.fixture(scope='module')
def other_mapping():
    """Disappearing middle carbon

    C C O

    C   C
    """
    molA = Chem.MolFromSmiles('CCO')
    molB = Chem.MolFromSmiles('CC')

    m = AtomMapping(molA, molB)
    m.AtoB[0] = 0
    m.AtoB[1] = None
    m.AtoB[2] = 1

    m.BtoA[0] = 0
    m.BtoA[1] = 2

    return m


def test_atommapping_usage(simple_mapping):
    assert len(simple_mapping.AtoB) == 3
    assert len(simple_mapping.BtoA) == 2

    assert simple_mapping.AtoB[1] == 1
    assert simple_mapping.AtoB[2] is None

    assert simple_mapping.BtoA[1] == 1
    with pytest.raises(KeyError):
        simple_mapping.AtoB[3]
    with pytest.raises(KeyError):
        simple_mapping.BtoA[2]


def test_atommapping_hash(simple_mapping, other_mapping):
    # these two mappings map the same molecules, but with a different mapping
    assert simple_mapping is not other_mapping
