# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import importlib
import string
import pytest
from importlib import resources
from rdkit import Chem
from openmm.app import PDBFile

import gufe
import openfe
from openfe.setup import SmallMoleculeComponent
from openfe.setup.atom_mapping import LigandAtomMapping

@pytest.fixture(scope='session')
def ethane():
    return SmallMoleculeComponent(Chem.MolFromSmiles('CC'))


@pytest.fixture(scope='session')
def simple_mapping():
    """Disappearing oxygen on end

    C C O

    C C
    """
    molA = SmallMoleculeComponent(Chem.MolFromSmiles('CCO'))
    molB = SmallMoleculeComponent(Chem.MolFromSmiles('CC'))

    m = LigandAtomMapping(molA, molB, molA_to_molB={0: 0, 1: 1})

    return m


@pytest.fixture(scope='session')
def other_mapping():
    """Disappearing middle carbon

    C C O

    C   C
    """
    molA = SmallMoleculeComponent(Chem.MolFromSmiles('CCO'))
    molB = SmallMoleculeComponent(Chem.MolFromSmiles('CC'))

    m = LigandAtomMapping(molA, molB, molA_to_molB={0: 0, 2: 1})

    return m


@pytest.fixture()
def lomap_basic_test_files_dir(tmpdir_factory):
    # for lomap, which wants the files in a directory
    lomap_files = tmpdir_factory.mktemp('lomap_files')
    lomap_basic = 'openfe.tests.data.lomap_basic'

    for f in importlib.resources.contents(lomap_basic):
        if not f.endswith('mol2'):
            continue
        stuff = importlib.resources.read_binary(lomap_basic, f)

        with open(str(lomap_files.join(f)), 'wb') as fout:
            fout.write(stuff)

    yield str(lomap_files)


@pytest.fixture(scope='session')
def lomap_basic_test_files():
    # a dict of {filenames.strip(mol2): SmallMoleculeComponent} for a simple
    # set of ligands
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
            mol = Chem.MolFromMol2File(str(fn), removeHs=False)
            files[f] = SmallMoleculeComponent(mol, name=f)

    return files


@pytest.fixture(scope='session')
def benzene_modifications():
    files = {}
    with importlib.resources.path('openfe.tests.data',
                                  'benzene_modifications.sdf') as fn:
        supp = Chem.SDMolSupplier(str(fn), removeHs=False)
        for rdmol in supp:
            files[rdmol.GetProp('_Name')] = SmallMoleculeComponent(rdmol)
    return files


@pytest.fixture
def serialization_template():
    def inner(filename):
        loc = "openfe.tests.data.serialization"
        tmpl = importlib.resources.read_text(loc, filename)
        return tmpl.format(GUFE_VERSION=gufe.__version__,
                           OFE_VERSION=openfe.__version__)

    return inner


@pytest.fixture(scope='session')
def benzene_transforms():
    # a dict of Molecules for benzene transformations
    mols = {}
    with resources.path('openfe.tests.data',
                        'benzene_modifications.sdf') as fn:
        supplier = Chem.SDMolSupplier(str(fn), removeHs=False)
        for mol in supplier:
            mols[mol.GetProp('_Name')] = SmallMoleculeComponent(mol)
    return mols


@pytest.fixture(scope='session')
def benzene_maps():
    MAPS = {
        'phenol': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
                   7: 7, 8: 8, 9: 9, 10: 12, 11: 11},
        'anisole': {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 10,
                    6: 11, 7: 12, 8: 13, 9: 14, 10: 2, 11: 15}}
    return MAPS


@pytest.fixture(scope='session')
def benzene_phenol_mapping(benzene_transforms, benzene_maps):
    molA = SmallMoleculeComponent(benzene_transforms['benzene'].to_rdkit())
    molB = SmallMoleculeComponent(benzene_transforms['phenol'].to_rdkit())
    m = LigandAtomMapping(molA, molB, benzene_maps['phenol'])
    return m


@pytest.fixture(scope='session')
def benzene_anisole_mapping(benzene_transforms, benzene_maps):
    molA = SmallMoleculeComponent(benzene_transforms['benzene'].to_rdkit())
    molB = SmallMoleculeComponent(benzene_transforms['anisole'].to_rdkit())
    m = LigandAtomMapping(molA, molB, benzene_maps['anisole'])
    return m


@pytest.fixture(scope='session')
def T4_protein_component():
    with resources.path('openfe.tests.data', '181l_only.pdb') as fn:
        comp = gufe.ProteinComponent.from_pdbfile(str(fn), name="T4_protein")

    return comp
