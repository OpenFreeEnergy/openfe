import pytest
from rdkit import Chem

from openfe.setup import AtomMapping
from openfe.setup import Molecule

@pytest.fixture(scope='session')
def ethane():
    return Molecule(Chem.MolFromSmiles('CC'))


@pytest.fixture(scope='session')
def simple_mapping():
    """Disappearing oxygen on end

    C C O

    C C
    """
    molA = Molecule(Chem.MolFromSmiles('CCO'))
    molB = Molecule(Chem.MolFromSmiles('CC'))

    m = AtomMapping(molA, molB, mol1_to_mol2={0: 0, 1: 1})

    return m


@pytest.fixture(scope='session')
def other_mapping():
    """Disappearing middle carbon

    C C O

    C   C
    """
    molA = Molecule(Chem.MolFromSmiles('CCO'))
    molB = Molecule(Chem.MolFromSmiles('CC'))

    m = AtomMapping(molA, molB, mol1_to_mol2={0: 0, 2: 1})

    return m
