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

    m = AtomMapping(molA, molB, mol1_to_mol2={0: 0, 1: 1})

    return m


@pytest.fixture(scope='module')
def other_mapping():
    """Disappearing middle carbon

    C C O

    C   C
    """
    molA = Chem.MolFromSmiles('CCO')
    molB = Chem.MolFromSmiles('CC')

    m = AtomMapping(molA, molB, mol1_to_mol2={0: 0, 2: 1})

    return m


def test_atommapping_usage(simple_mapping):
    assert simple_mapping.mol1_to_mol2[1] == 1
    assert simple_mapping.mol1_to_mol2.get(2, None) is None

    with pytest.raises(KeyError):
        simple_mapping.mol1_to_mol2[3]


def test_atommapping_hash(simple_mapping, other_mapping):
    # these two mappings map the same molecules, but with a different mapping
    assert simple_mapping is not other_mapping
