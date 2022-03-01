import importlib
import importlib.resources
import openff.toolkit.topology
import pytest

try:
    from openeye import oechem
except ImportError:
    HAS_OECHEM = False
else:
    HAS_OECHEM = oechem.OEChemIsLicensed()
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


@pytest.mark.parametrize('internal,rdkit_name,name,expected', [
    ('', 'foo', '', 'foo'),
    ('', '', 'foo', 'foo'),
    ('', 'bar', 'foo', 'foo'),
    ('bar', '', 'foo', 'foo'),
    ('baz', 'bar', 'foo', 'foo'),
    ('foo', '', '', 'foo'),
])
def test_ensure_ofe_name(internal, rdkit_name, name, expected, recwarn):
    rdkit = Chem.MolFromSmiles("CC")
    if internal:
        rdkit.SetProp('_Name', internal)

    if rdkit_name:
        rdkit.SetProp('ofe-name', rdkit_name)

    out_name = _ensure_ofe_name(rdkit, name)

    if {rdkit_name, internal} - {'foo', ''}:
        # we should warn if rdkit properties are anything other than 'foo'
        # (expected) or the empty string (not set)
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
        assert ethane.to_rdkit() is not alt_ethane.to_rdkit()

    def test_rdkit_independence(self):
        # once we've constructed a Molecule, it is independent from the source
        mol = Chem.MolFromSmiles('CC')
        our_mol = Molecule.from_rdkit(mol)

        mol.SetProp('foo', 'bar')  # this is the source molecule, not ours
        with pytest.raises(KeyError):
            our_mol.to_rdkit().GetProp('foo')

    def test_rdkit_copy_source_copy(self):
        # we should copy in any properties that were in the source molecule
        mol = Chem.MolFromSmiles('CC')
        mol.SetProp('foo', 'bar')
        our_mol = Molecule.from_rdkit(mol)

        assert our_mol.to_rdkit().GetProp('foo') == 'bar'

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

    def test_to_sdf_string(self, named_ethane, serialization_template):
        expected = serialization_template("ethane_template.sdf")
        assert named_ethane.to_sdf() == expected

    def test_from_sdf_string(self, named_ethane, serialization_template):
        sdf_str = serialization_template("ethane_template.sdf")
        assert Molecule.from_sdf_string(sdf_str) == named_ethane

    def test_from_sdf_file(self, named_ethane, serialization_template,
                           tmpdir):
        sdf_str = serialization_template("ethane_template.sdf")
        with open(tmpdir / "temp.sdf", mode='w') as tmpf:
            tmpf.write(sdf_str)

        assert Molecule.from_sdf_file(tmpdir / "temp.sdf") == named_ethane

    def test_from_sdf_file_junk(self):
        with importlib.resources.path('openfe.tests.data.lomap_basic',
                                      'toluene.mol2') as fn:
            with pytest.raises(ValueError):
                Molecule.from_sdf_file(str(fn))

    def test_from_sdf_string_multiple_molecules(self):
        contents = importlib.resources.read_text("openfe.tests.data",
                                                 "multi_molecule.sdf")
        with pytest.raises(RuntimeError, match="contains more than 1"):
            Molecule.from_sdf_string(contents)

    def test_from_rdkit(self, named_ethane):
        rdkit = Chem.MolFromSmiles("CC")
        mol = Molecule.from_rdkit(rdkit, "ethane")
        assert mol == named_ethane
        assert mol.to_rdkit() is not rdkit


class TestMoleculeConversion:
    def test_to_off(self, ethane):
        off_ethane = ethane.to_openff()

        assert isinstance(off_ethane, openff.toolkit.topology.Molecule)

    def test_to_off_name(self, named_ethane):
        off_ethane = named_ethane.to_openff()

        assert off_ethane.name == 'ethane'

    @pytest.mark.skipif(not HAS_OECHEM, reason="No OEChem available")
    def test_to_oechem(self, ethane):
        oec_ethane = ethane.to_openeye()

        assert isinstance(oec_ethane, oechem.OEMol)
