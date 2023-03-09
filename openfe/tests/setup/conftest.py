# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import importlib
import pytest
from importlib import resources
from rdkit import Chem
from rdkit.Chem import AllChem

import gufe
import openfe
from gufe import SmallMoleculeComponent, LigandAtomMapping


def mol_from_smiles(smiles: str) -> Chem.Mol:
    m = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(m)

    return m

@pytest.fixture(scope='session')
def ethane():
    return SmallMoleculeComponent(mol_from_smiles('CC'))


@pytest.fixture(scope='session')
def simple_mapping():
    """Disappearing oxygen on end

    C C O

    C C
    """
    molA = SmallMoleculeComponent(mol_from_smiles('CCO'))
    molB = SmallMoleculeComponent(mol_from_smiles('CC'))

    m = LigandAtomMapping(molA, molB, componentA_to_componentB={0: 0, 1: 1})

    return m


@pytest.fixture(scope='session')
def other_mapping():
    """Disappearing middle carbon

    C C O

    C   C
    """
    molA = SmallMoleculeComponent(mol_from_smiles('CCO'))
    molB = SmallMoleculeComponent(mol_from_smiles('CC'))

    m = LigandAtomMapping(molA, molB, componentA_to_componentB={0: 0, 2: 1})

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
def atom_mapping_basic_test_files():
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
        return tmpl.replace('{OFE_VERSION}', openfe.__version__)

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
def T4_protein_component():
    with resources.path('openfe.tests.data', '181l_only.pdb') as fn:
        comp = gufe.ProteinComponent.from_pdb_file(str(fn), name="T4_protein")

    return comp
