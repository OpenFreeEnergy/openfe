# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import os
import importlib
import pytest
from importlib import resources
from rdkit import Chem
from rdkit.Chem import AllChem
from openff.units import unit

import gufe
import openfe
from gufe import SmallMoleculeComponent, LigandAtomMapping


class SlowTests:
    """Plugin for handling fixtures that skips slow tests

    Fixtures
    --------

    Currently two fixture types are handled:
      * `integration`:
        Extremely slow tests that are meant to be run to truly put the code
        through a real run.

      * `slow`:
        Unit tests that just take too long to be running regularly.


    How to use the fixtures
    -----------------------

    To add these fixtures simply add `@pytest.mark.integration` or
    `@pytest.mark.slow` decorator to the relevant function or class.


    How to run tests marked by these fixtures
    -----------------------------------------

    To run the `integration` tests, either use the `--integration` flag
    when invoking pytest, or set the environment variable
    `OFE_INTEGRATION_TESTS` to `true`. Note: triggering `integration` will
    automatically also trigger tests marked as `slow`.

    To run the `slow` tests, either use the `--runslow` flag when invoking
    pytest, or set the environment variable `OFE_SLOW_TESTS` to `true`
    """
    def __init__(self, config):
        self.config = config

    @staticmethod
    def _modify_slow(items, config):
        msg = ("need --runslow pytest cli option or the environment variable "
               "`OFE_SLOW_TESTS` set to `True` to run")
        skip_slow = pytest.mark.skip(reason=msg)
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    @staticmethod
    def _modify_integration(items, config):
        msg = ("need --integration pytest cli option or the environment "
               "variable `OFE_INTEGRATION_TESTS` set to `True` to run")
        skip_int = pytest.mark.skip(reason=msg)
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_int)

    def pytest_collection_modifyitems(self, items, config):
        if (config.getoption('--integration') or
            os.getenv("OFE_INTEGRATION_TESTS", default="false").lower() == 'true'):
            return
        elif (config.getoption('--runslow') or
              os.getenv("OFE_SLOW_TESTS", default="false").lower() == 'true'):
            self._modify_integration(items, config)
        else:
            self._modify_integration(items, config)
            self._modify_slow(items, config)


# allow for optional slow tests
# See: https://docs.pytest.org/en/latest/example/simple.html
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--integration", action="store_true", default=False,
        help="run long integration tests",
    )


def pytest_configure(config):
    config.pluginmanager.register(SlowTests(config), "slow")
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line(
            "markers", "integration: mark test as long integration test")


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
        with importlib.resources.files('openfe.tests.data.lomap_basic') as d:
            fn = str(d / (f + '.mol2'))
            mol = Chem.MolFromMol2File(fn, removeHs=False)
            files[f] = SmallMoleculeComponent(mol, name=f)

    return files


@pytest.fixture(scope='session')
def benzene_modifications():
    files = {}
    with importlib.resources.files('openfe.tests.data') as d:
        fn = str(d / 'benzene_modifications.sdf')
        supp = Chem.SDMolSupplier(str(fn), removeHs=False)
        for rdmol in supp:
            files[rdmol.GetProp('_Name')] = SmallMoleculeComponent(rdmol)
    return files


@pytest.fixture(scope='session')
def charged_benzene_modifications():
    files = {}
    with importlib.resources.files('openfe.tests.data.openmm_rfe') as d:
        fn = str(d / 'charged_benzenes.sdf')
        supp = Chem.SDMolSupplier(str(fn), removeHs=False)
        for rdmol in supp:
            files[rdmol.GetProp('_Name')] = SmallMoleculeComponent(rdmol)
    return files


@pytest.fixture(scope='session')
def bace_ligands():
    files = {}
    with importlib.resources.files('openfe.tests.data.openmm_septop') as d:
        fn = str(d / 'bace1.sdf')
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
    with resources.files('openfe.tests.data') as d:
        fn = str(d / 'benzene_modifications.sdf')
        supplier = Chem.SDMolSupplier(fn, removeHs=False)
        for mol in supplier:
            mols[mol.GetProp('_Name')] = SmallMoleculeComponent(mol)
    return mols


@pytest.fixture(scope='session')
def T4_protein_component():
    with resources.files('openfe.tests.data') as d:
        fn = str(d / '181l_only.pdb')
        comp = gufe.ProteinComponent.from_pdb_file(fn, name="T4_protein")

    return comp

@pytest.fixture(scope='session')
def bace_protein_component():
    with resources.files('openfe.tests.data.openmm_septop') as d:
        fn = str(d / 'bace.pdb')
        comp = gufe.ProteinComponent.from_pdb_file(fn, name="BACE")

    return comp


@pytest.fixture(scope='session')
def eg5_protein_pdb():
    with resources.files('openfe.tests.data.eg5') as d:
        yield str(d / 'eg5_protein.pdb')


@pytest.fixture()
def eg5_ligands_sdf():
    with resources.files('openfe.tests.data.eg5') as d:
        yield str(d / 'eg5_ligands.sdf')


@pytest.fixture()
def eg5_cofactor_sdf():
    with resources.files('openfe.tests.data.eg5') as d:
        yield str(d / 'eg5_cofactor.sdf')


@pytest.fixture()
def eg5_protein(eg5_protein_pdb) -> openfe.ProteinComponent:
    return openfe.ProteinComponent.from_pdb_file(eg5_protein_pdb)


@pytest.fixture()
def eg5_ligands(eg5_ligands_sdf) -> list[SmallMoleculeComponent]:
    return [SmallMoleculeComponent(m)
            for m in Chem.SDMolSupplier(eg5_ligands_sdf, removeHs=False)]


@pytest.fixture()
def eg5_cofactor(eg5_cofactor_sdf) -> SmallMoleculeComponent:
    return SmallMoleculeComponent.from_sdf_file(eg5_cofactor_sdf)


@pytest.fixture()
def orion_network():
    with resources.files('openfe.tests.data.external_formats') as d:
        yield str(d / 'somebenzenes_nes.dat')


@pytest.fixture()
def fepplus_network():
    with resources.files('openfe.tests.data.external_formats') as d:
        yield str(d / 'somebenzenes_edges.edge')


@pytest.fixture()
def CN_molecule():
    """
    A basic CH3NH2 molecule for quick testing.
    """
    with resources.files('openfe.tests.data') as d:
        fn = str(d / 'CN.sdf')
        supp = Chem.SDMolSupplier(str(fn), removeHs=False)

        smc = [SmallMoleculeComponent(i) for i in supp][0]

    return smc


@pytest.fixture(scope='function')
def am1bcc_ref_charges():
    ref_chgs = {
        'ambertools': [
            0.146957, -0.918943, 0.025557, 0.025557,
            0.025557, 0.347657, 0.347657
        ] * unit.elementary_charge,
        'openeye': [
            0.14713, -0.92016, 0.02595, 0.02595,
            0.02595, 0.34759, 0.34759
        ] * unit.elementary_charge,
        'nagl': [
            0.170413, -0.930417, 0.021593, 0.021593,
            0.021593, 0.347612, 0.347612
        ] * unit.elementary_charge,
        'espaloma': [
            0.017702, -0.966793, 0.063076, 0.063076,
            0.063076, 0.379931, 0.379931
        ] * unit.elementary_charge,
    }
    return ref_chgs
