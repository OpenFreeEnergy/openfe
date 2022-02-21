import openff.toolkit.topology
import pytest

try:
    from openeye import oechem
except ImportError:
    HAS_OECHEM = False
else:
    HAS_OECHEM = oechem.OEChemIsLicensed()
from openfe.setup import Molecule
from rdkit import Chem


@pytest.fixture
def alt_ethane():
    return Molecule(Chem.MolFromSmiles("CC"))


@pytest.fixture
def named_ethane():
    mol = Chem.MolFromSmiles("CC")

    return Molecule(mol, name='ethane')


class TestMolecule:
    def test_rdkit_behavior(self, ethane, alt_ethane):
        # Check that fixture setup is correct (we aren't accidentally
        # testing tautologies)
        assert ethane is not alt_ethane
        assert ethane.rdkit is not alt_ethane.rdkit

    def test_equality_and_hash(self, ethane, alt_ethane):
        assert hash(ethane) == hash(alt_ethane)
        assert ethane == alt_ethane

    def test_equality_and_hash_name_differs(self, ethane, named_ethane):
        # names would be used to distinguish different binding modes
        assert hash(ethane) != hash(named_ethane)
        assert ethane != named_ethane

    def test_smiles(self, named_ethane):
        assert named_ethane.smiles == 'CC'

    def test_name(self, named_ethane):
        assert named_ethane.name == 'ethane'

    def test_empty_name(self, alt_ethane):
        assert alt_ethane.name == ''


class TestMoleculeConversion:
    def test_to_off(self, ethane):
        off_ethane = ethane.openff

        assert isinstance(off_ethane, openff.toolkit.topology.Molecule)

    def test_to_off_name(self, named_ethane):
        off_ethane = named_ethane.openff

        assert off_ethane.name == 'ethane'

    @pytest.mark.skipif(not HAS_OECHEM, reason="No OEChem available")
    def test_to_oechem(self, ethane):
        if not HAS_OECHEM:
            pytest.skip()
        oec_ethane = ethane.oechem

        assert isinstance(oec_ethane, oechem.OEMol)
