import pytest

from openfe.setup import Molecule
from rdkit import Chem


@pytest.fixture
def alt_ethane():
    return Molecule(Chem.MolFromSmiles("CC"))


@pytest.fixture
def named_ethane():
    mol = Chem.MolFromSmiles("CC")

    return Molecule(mol, name='ethane')


def test_ensure_ofe_name():
    pytest.skip()


def test_ensure_ofe_version():
    pytest.skip()


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

    def test_serialization_cycle(self, named_ethane):
        serialized = named_ethane.to_sdf()
        deserialized = Molecule.from_sdf_string(serialized)
        reserialized = deserialized.to_sdf()

        assert named_ethane == deserialized
        assert serialized == reserialized

    def test_to_sdf_string(self, named_ethane):
        pytest.skip()

    def test_from_sdf_string(self):
        pytest.skip()
