# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import importlib
import importlib.resources
import string
import pytest
from rdkit import Chem

import openfe
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


@pytest.fixture(scope='session')
def lomap_basic_test_files():
    # a dict of {filenames.strip(mol2): Molecule} for a simple set of ligands
    files = {}
    for f in [
        '1,3,7-trimethylnaphthalene',
        '1-butyl-4-methylbenzene',
        '2,6-dimethylnaphthalene',
        '2-methyl-6-propylnaphthalene',
        '2-methylnaphthalene',
        '2-naftanol',
        'methylcyclohexane',
        'toluene']:
        with importlib.resources.path('openfe.tests.data.lomap_basic',
                                      f + '.mol2') as fn:
            mol = Chem.MolFromMol2File(str(fn))
            files[f] = Molecule(mol, name=f)

    return files


@pytest.fixture
def serialization_template():
    def inner(filename):
        loc = "openfe.tests.data.serialization"
        tmpl = importlib.resources.read_text(loc, filename)
        return tmpl.format(OFE_VERSION=openfe.__version__)

    return inner


@pytest.fixture
def PDB_181L_path():
    with importlib.resources.path('openfe.tests.data', '181l.pdb') as f:
        yield str(f)


@pytest.fixture
def PDBx_181L_path():
    with importlib.resources.path('openfe.tests.data', '181l.cif') as f:
        yield str(f)
