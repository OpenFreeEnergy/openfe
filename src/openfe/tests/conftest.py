# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import gzip
import os
import pathlib
import urllib.error
import urllib.request
from importlib import resources

import gufe
import mdtraj
import numpy as np
import openmm
import pandas as pd
import pooch
import pytest
from gufe import AtomMapper, LigandAtomMapping, ProteinComponent, SmallMoleculeComponent
from openff.toolkit import ForceField
from openff.units import unit as offunit
from openmm import unit as ommunit
from rdkit import Chem
from rdkit.Chem import AllChem

import openfe
from openfe.data._registry import POOCH_CACHE
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
from openfe.protocols.openmm_rfe._rfe_utils.relative import HybridTopologyFactory
from openfe.protocols.openmm_utils.serialization import deserialize
from openfe.tests.protocols.openmm_rfe.helpers import make_htf


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
        msg = (
            "need --runslow pytest cli option or the environment variable "
            "`OFE_SLOW_TESTS` set to `True` to run"
        )
        skip_slow = pytest.mark.skip(reason=msg)
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    @staticmethod
    def _modify_integration(items, config):
        msg = (
            "need --integration pytest cli option or the environment "
            "variable `OFE_INTEGRATION_TESTS` set to `True` to run"
        )
        skip_int = pytest.mark.skip(reason=msg)
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_int)

    def pytest_collection_modifyitems(self, items, config):
        if (
            config.getoption("--integration")
            or os.getenv("OFE_INTEGRATION_TESTS", default="false").lower() == "true"
        ):
            return
        elif (
            config.getoption("--runslow")
            or os.getenv("OFE_SLOW_TESTS", default="false").lower() == "true"
        ):
            self._modify_integration(items, config)
        else:
            self._modify_integration(items, config)
            self._modify_slow(items, config)


# allow for optional slow tests
# See: https://docs.pytest.org/en/latest/example/simple.html
def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--integration", action="store_true", default=False, help="run long integration tests")  # fmt: skip


def pytest_configure(config):
    config.pluginmanager.register(SlowTests(config), "slow")
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "integration: mark test as long integration test")


def mol_from_smiles(smiles: str) -> Chem.Mol:
    m = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(m)  # type: ignore[attr-defined]

    return m


@pytest.fixture(scope="session")
def ethane():
    return SmallMoleculeComponent(mol_from_smiles("CC"))


@pytest.fixture(scope="session")
def simple_mapping():
    """Disappearing oxygen on end

    C C O

    C C
    """
    molA = SmallMoleculeComponent(mol_from_smiles("CCO"))
    molB = SmallMoleculeComponent(mol_from_smiles("CC"))

    m = LigandAtomMapping(molA, molB, componentA_to_componentB={0: 0, 1: 1})

    return m


@pytest.fixture(scope="session")
def other_mapping():
    """Disappearing middle carbon

    C C O

    C   C
    """
    molA = SmallMoleculeComponent(mol_from_smiles("CCO"))
    molB = SmallMoleculeComponent(mol_from_smiles("CC"))

    m = LigandAtomMapping(molA, molB, componentA_to_componentB={0: 0, 2: 1})

    return m


@pytest.fixture()
def lomap_basic_test_files_dir(tmpdir_factory):
    # for lomap, which wants the files in a directory
    lomap_files = tmpdir_factory.mktemp("lomap_files")
    lomap_basic = "openfe.tests.data.lomap_basic"

    for f in resources.contents(lomap_basic):
        if not f.endswith("mol2"):
            continue
        stuff = resources.read_binary(lomap_basic, f)

        with open(str(lomap_files.join(f)), "wb") as fout:
            fout.write(stuff)

    yield str(lomap_files)


@pytest.fixture(scope="session")
def atom_mapping_basic_test_files():
    # a dict of {filenames.strip(mol2): SmallMoleculeComponent} for a simple
    # set of ligands
    files = {}
    for f in [
        "1,3,7-trimethylnaphthalene",
        "1-butyl-4-methylbenzene",
        "2,6-dimethylnaphthalene",
        "2-methyl-6-propylnaphthalene",
        "2-methylnaphthalene",
        "2-naftanol",
        "methylcyclohexane",
        "toluene",
    ]:
        with resources.as_file(resources.files("openfe.tests.data.lomap_basic")) as d:
            fn = str(d / (f + ".mol2"))
            mol = Chem.MolFromMol2File(fn, removeHs=False)
            files[f] = SmallMoleculeComponent(mol, name=f)

    return files


@pytest.fixture()
def lomap_old_mapper() -> AtomMapper:
    """
    LomapAtomMapper with the old default settings.

    This is necessary as atom_mapping_basic_test_files are not all fully aligned
    and need both shift and a large max3d value.
    """
    return openfe.setup.atom_mapping.LomapAtomMapper(
        time=20,
        threed=True,
        max3d=1000.0,
        element_change=True,
        seed="",
        shift=True,
    )


@pytest.fixture
def benzene_toluene_topology():
    """Load the mdtraj hybrid topology reference for benzene to toluene."""
    with resources.as_file(resources.files("openfe.tests.data.openmm_rfe")) as d:
        atoms = pd.read_csv(d / "benzene_toluene_hybrid_top" / "hybrid_topology_atoms.csv")
        bonds = np.loadtxt(d / "benzene_toluene_hybrid_top" / "hybrid_topology_bonds.txt")
        return mdtraj.Topology.from_dataframe(atoms=atoms, bonds=bonds)


@pytest.fixture(scope="session")
def benzene_modifications():
    files = {}
    with resources.as_file(resources.files("openfe.tests.data")) as d:
        fn = str(d / "benzene_modifications.sdf")
        supp = Chem.SDMolSupplier(str(fn), removeHs=False)
        for rdmol in supp:
            files[rdmol.GetProp("_Name")] = SmallMoleculeComponent(rdmol)
    return files


@pytest.fixture(scope="session")
def charged_benzene_modifications():
    files = {}
    with resources.as_file(resources.files("openfe.tests.data.openmm_rfe")) as d:
        fn = str(d / "charged_benzenes.sdf")
        supp = Chem.SDMolSupplier(str(fn), removeHs=False)
        for rdmol in supp:
            files[rdmol.GetProp("_Name")] = SmallMoleculeComponent(rdmol)
    return files


@pytest.fixture(scope="session")
def T4L_septop_reference_xml():
    with resources.as_file(resources.files("openfe.tests.data.openmm_septop")) as d:
        f = str(d / "system.xml.bz2")
    return deserialize(pathlib.Path(f))


@pytest.fixture
def serialization_template():
    def inner(filename):
        loc = "openfe.tests.data.serialization"
        tmpl = resources.read_text(loc, filename)
        return tmpl.replace("{OFE_VERSION}", openfe.__version__)

    return inner


@pytest.fixture(scope="session")
def benzene_transforms():
    # a dict of Molecules for benzene transformations
    mols = {}
    with resources.as_file(resources.files("openfe.tests.data")) as d:
        fn = str(d / "benzene_modifications.sdf")
        supplier = Chem.SDMolSupplier(fn, removeHs=False)
        for mol in supplier:
            mols[mol.GetProp("_Name")] = SmallMoleculeComponent(mol)
    return mols


@pytest.fixture(scope="session")
def T4_protein_component():
    with resources.as_file(resources.files("openfe.tests.data")) as d:
        fn = str(d / "181l_only.pdb")

    return gufe.ProteinComponent.from_pdb_file(fn, name="T4_protein")


@pytest.fixture(scope="session")
def a2a_protein_membrane_component():
    with resources.as_file(resources.files("openfe.tests.data")) as d:
        with gzip.open(d / "a2a/protein.pdb.gz", "rb") as f:
            yield openfe.ProteinMembraneComponent.from_pdb_file(f, name="a2a")


@pytest.fixture(scope="session")
def eg5_protein_pdb():
    with resources.as_file(resources.files("openfe.tests.data.eg5")) as d:
        yield str(d / "eg5_protein.pdb")


@pytest.fixture()
def eg5_ligands_sdf():
    with resources.as_file(resources.files("openfe.tests.data.eg5")) as d:
        yield str(d / "eg5_ligands.sdf")


@pytest.fixture()
def eg5_cofactor_sdf():
    with resources.as_file(resources.files("openfe.tests.data.eg5")) as d:
        yield str(d / "eg5_cofactor.sdf")


@pytest.fixture()
def eg5_protein(eg5_protein_pdb) -> openfe.ProteinComponent:
    return openfe.ProteinComponent.from_pdb_file(eg5_protein_pdb)


@pytest.fixture()
def eg5_ligands(eg5_ligands_sdf) -> list[SmallMoleculeComponent]:
    return [SmallMoleculeComponent(m) for m in Chem.SDMolSupplier(eg5_ligands_sdf, removeHs=False)]


@pytest.fixture()
def eg5_cofactor(eg5_cofactor_sdf) -> SmallMoleculeComponent:
    return SmallMoleculeComponent.from_sdf_file(eg5_cofactor_sdf)


@pytest.fixture(scope="session")
def a2a_ligands_sdf():
    with resources.as_file(resources.files("openfe.tests.data.a2a")) as d:
        yield str(d / "ligands.sdf.gz")


@pytest.fixture(scope="session")
def a2a_ligands(a2a_ligands_sdf):
    with gzip.open(a2a_ligands_sdf, "rb") as gzf:
        suppl = Chem.ForwardSDMolSupplier(gzf, removeHs=False)
        yield [SmallMoleculeComponent(m) for m in suppl]


@pytest.fixture()
def orion_network():
    with resources.as_file(resources.files("openfe.tests.data.external_formats")) as d:
        yield str(d / "somebenzenes_nes.dat")


@pytest.fixture()
def fepplus_network():
    with resources.as_file(resources.files("openfe.tests.data.external_formats")) as d:
        yield str(d / "somebenzenes_edges.edge")


@pytest.fixture()
def CN_molecule():
    """
    A basic CH3NH2 molecule for quick testing.
    """
    with resources.as_file(resources.files("openfe.tests.data")) as d:
        fn = str(d / "CN.sdf")
        supp = Chem.SDMolSupplier(str(fn), removeHs=False)

        smc = [SmallMoleculeComponent(i) for i in supp][0]

    return smc


@pytest.fixture(scope="function")
def am1bcc_ref_charges():
    ref_chgs = {
        "ambertools":[
            0.146957, -0.918943, 0.025557, 0.025557,
            0.025557, 0.347657, 0.347657
        ] * offunit.elementary_charge,
        "openeye": [
            0.14713, -0.92016, 0.02595, 0.02595,
            0.02595, 0.34759, 0.34759
        ] * offunit.elementary_charge,
        "nagl": [
            0.170413, -0.930417, 0.021593, 0.021593,
            0.021593, 0.347612, 0.347612
        ] * offunit.elementary_charge,
        "espaloma": [
            0.017702, -0.966793, 0.063076, 0.063076,
            0.063076, 0.379931, 0.379931
        ] * offunit.elementary_charge,
    }  # fmt: skip
    return ref_chgs


try:
    urllib.request.urlopen("https://www.google.com")
except urllib.error.URLError:  # -no-cov-
    HAS_INTERNET = False
else:
    HAS_INTERNET = True

try:
    import espaloma

    HAS_ESPALOMA = True
except ModuleNotFoundError:
    HAS_ESPALOMA = False


@pytest.fixture(scope="module")
def chlorobenzene():
    """Load chlorobenzene with partial charges from sdf file."""
    with resources.as_file(resources.files("openfe.tests.data.htf")) as f:
        yield SmallMoleculeComponent.from_sdf_file(f / "t4_lysozyme_data" / "chlorobenzene.sdf")


@pytest.fixture(scope="module")
def fluorobenzene():
    """Load fluorobenzene with partial charges from sdf file."""
    with resources.as_file(resources.files("openfe.tests.data.htf")) as f:
        yield SmallMoleculeComponent.from_sdf_file(f / "t4_lysozyme_data" / "fluorobenzene.sdf")


@pytest.fixture(scope="module")
def chlorobenzene_to_fluorobenzene_mapping(chlorobenzene, fluorobenzene):
    """Return a mapping from chlorobenzene to fluorobenzene."""
    return LigandAtomMapping(
        componentA=chlorobenzene,
        componentB=fluorobenzene,
        componentA_to_componentB={
            # perfect one-to-one mapping
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
            7: 7,
            8: 8,
            9: 9,
            10: 10,
            11: 11,
        },
    )


@pytest.fixture(scope="module")
def t4_lysozyme_solvated():
    """Load the T4 lysozyme L99A structure and solvent from the pdb file."""
    with resources.as_file(resources.files("openfe.tests.data.htf")) as f:
        with gzip.open(f / "t4_lysozyme_data" / "t4_lysozyme_solvated.pdb.gz", "rb") as gzf:
            yield ProteinComponent.from_pdb_file(gzf)


def apply_box_vectors_and_fix_nb_force(
    hybrid_topology_factory: HybridTopologyFactory, force_field: ForceField
):
    """
    Edit the systems in the hybrid topology factory to have the correct box vectors and nonbonded force settings for the T4 lysozyme system.
    """
    hybrid_system = hybrid_topology_factory.hybrid_system
    # as we use a pre-solvated system, we need to correct the nonbonded methods and cutoffs and set the box vectors
    box_vectors = [
        openmm.vec3.Vec3(x=6.90789161545809, y=0.0, z=0.0) * ommunit.nanometer,
        openmm.vec3.Vec3(x=0.0, y=6.90789161545809, z=0.0) * ommunit.nanometer,
        openmm.vec3.Vec3(x=3.453945807729045, y=3.453945807729045, z=4.88461700499211)
        * ommunit.nanometer,
    ]
    hybrid_system.setDefaultPeriodicBoxVectors(*box_vectors)
    for force in hybrid_system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            force.setNonbondedMethod(openmm.NonbondedForce.PME)
            force.setCutoffDistance(
                force_field.get_parameter_handler("Electrostatics").cutoff.m_as(offunit.nanometer)
                * ommunit.nanometer
            )
            force.setUseDispersionCorrection(False)
            force.setUseSwitchingFunction(False)
        elif isinstance(force, openmm.CustomNonbondedForce):
            force.setCutoffDistance(
                force_field.get_parameter_handler("Electrostatics").cutoff.m_as(offunit.nanometer)
                * ommunit.nanometer
            )
            force.setNonbondedMethod(force.CutoffPeriodic)
            force.setUseLongRangeCorrection(False)
            force.setUseSwitchingFunction(False)

    # make sure both end state systems have the same cutoff method and distance
    for end_state in [hybrid_topology_factory._old_system, hybrid_topology_factory._new_system]:
        end_state.setDefaultPeriodicBoxVectors(*box_vectors)
        for force in end_state.getForces():
            if isinstance(force, openmm.NonbondedForce):
                force.setNonbondedMethod(openmm.NonbondedForce.PME)
                force.setCutoffDistance(
                    force_field.get_parameter_handler("Electrostatics").cutoff.m_as(
                        offunit.nanometer
                    )
                    * ommunit.nanometer
                )
                force.setUseDispersionCorrection(False)
                force.setUseSwitchingFunction(False)


@pytest.fixture(scope="module")
def htf_cmap_chlorobenzene_to_fluorobenzene(
    chlorobenzene_to_fluorobenzene_mapping, t4_lysozyme_solvated
):
    """Generate the htf for chlorobenzene to fluorobenzene with a CMAP term."""
    settings = RelativeHybridTopologyProtocol.default_settings()
    # make sure we interpolate the 1-4 exceptions involving dummy atoms if present
    settings.alchemical_settings.turn_off_core_unique_exceptions = True
    small_ff = settings.forcefield_settings.small_molecule_forcefield
    if ".offxml" not in small_ff:
        small_ff += ".offxml"
    ff = ForceField(small_ff)
    # update the default force fields to include a force field with CMAP terms
    settings.forcefield_settings.forcefields = [
        "amber/protein.ff19SB.xml",  # cmap amber ff
        "amber/tip3p_standard.xml",  # TIP3P and recommended monovalent ion parameters
        "amber/tip3p_HFE_multivalent.xml",  # for divalent ions
        "amber/phosaa19SB.xml",  # Handles THE TPO
    ]
    htf = make_htf(
        mapping=chlorobenzene_to_fluorobenzene_mapping,
        protein=t4_lysozyme_solvated,
        settings=settings,
    )
    # apply box vectors and fix nonbonded force settings so we can use PME
    apply_box_vectors_and_fix_nb_force(hybrid_topology_factory=htf, force_field=ff)
    hybrid_system = htf.hybrid_system
    forces = {force.getName(): force for force in hybrid_system.getForces()}

    return {
        "htf": htf,
        "hybrid_system": hybrid_system,
        "forces": forces,
        "mapping": chlorobenzene_to_fluorobenzene_mapping,
        "chlorobenzene": chlorobenzene_to_fluorobenzene_mapping.componentA,
        "fluorobenzene": chlorobenzene_to_fluorobenzene_mapping.componentB,
        "electrostatic_scale": ff.get_parameter_handler("Electrostatics").scale14,
        "vdW_scale": ff.get_parameter_handler("vdW").scale14,
        "force_field": ff,
    }
