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

    m = AtomMapping(molA, molB,
                    mol1_to_mol2={0: 0, 1: 1, 2: None},
                    mol2_to_mol1={0: 0, 1: 1})

    return m


@pytest.fixture(scope='module')
def other_mapping():
    """Disappearing middle carbon

    C C O

    C   C
    """
    molA = Chem.MolFromSmiles('CCO')
    molB = Chem.MolFromSmiles('CC')

    m = AtomMapping(molA, molB,
                    mol1_to_mol2={0: 0, 1: None, 2: 1},
                    mol2_to_mol1={0: 0, 1: 2})

    return m


def test_atommapping_usage(simple_mapping):
    assert len(simple_mapping.mol1_to_mol2) == 3
    assert len(simple_mapping.mol2_to_mol1) == 2

    assert simple_mapping.mol1_to_mol2[1] == 1
    assert simple_mapping.mol1_to_mol2[2] is None

    assert simple_mapping.mol2_to_mol1[1] == 1
    with pytest.raises(KeyError):
        simple_mapping.mol1_to_mol2[3]
    with pytest.raises(KeyError):
        simple_mapping.mol2_to_mol1[2]


def test_atommapping_hash(simple_mapping, other_mapping):
    # these two mappings map the same molecules, but with a different mapping
    assert simple_mapping is not other_mapping
