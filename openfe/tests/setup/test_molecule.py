import pytest

from openfe.setup import Molecule
from rdkit import Chem

@pytest.fixture
def alt_ethane():
    return Molecule(Chem.MolFromSmiles("CC"))

@pytest.fixture
def named_ethane():
    mol = Chem.MolFromSmiles("CC")
    mol.SetProp("_Name", "ethane")
    return Molecule(mol)

class TestMolecule:
    def test_rdkit_behavior(self, ethane, alt_ethane):
        # Check that fixture setup is correct (we aren't accidentally
        # testing tautologies) and that rdkit continues to behave in a way
        # that require our custom equality.
        assert ethane is not alt_ethane
        assert ethane.rdkit is not alt_ethane.rdkit
        assert ethane.rdkit != alt_ethane.rdkit  # rdkit might change this

    def test_equality_and_hash(self, ethane, alt_ethane):
        assert hash(ethane) == hash(alt_ethane)
        assert ethane == alt_ethane

    def test_equality_and_hash_name_differs(self, ethane, named_ethane):
        # names would be used to distinguish different binding modes
        assert hash(ethane) != hash(named_ethane)
        assert ethane != named_ethane

