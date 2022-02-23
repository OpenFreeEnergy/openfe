import pytest

from openfe.setup import Molecule
from openfe.setup.molecule import _ensure_ofe_name, _ensure_ofe_version
import openfe
from rdkit import Chem


@pytest.fixture
def alt_ethane():
    return Molecule(Chem.MolFromSmiles("CC"))


@pytest.fixture
def named_ethane():
    mol = Chem.MolFromSmiles("CC")

    return Molecule(mol, name='ethane')


@pytest.mark.parametrize('rdkit_name,name,expected', [
    ('foo', '', 'foo'),
    ('', 'foo', 'foo'),
    ('bar', 'foo', 'foo'),
])
def test_ensure_ofe_name(rdkit_name, name, expected, recwarn):
    rdkit = Chem.MolFromSmiles("CC")
    if rdkit_name:
        rdkit.SetProp('ofe-name', rdkit_name)

    out_name = _ensure_ofe_name(rdkit, name)

    if rdkit_name == "bar":
        assert len(recwarn) == 1
        assert "Molecule being renamed" in recwarn[0].message.args[0]
    else:
        assert len(recwarn) == 0

    assert out_name == expected
    assert rdkit.GetProp("ofe-name") == out_name


def test_ensure_ofe_version():
    rdkit = Chem.MolFromSmiles("CC")
    _ensure_ofe_version(rdkit)
    assert rdkit.GetProp("ofe-version") == openfe.__version__


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
