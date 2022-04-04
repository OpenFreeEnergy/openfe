import pytest
import pathlib
import json
from rdkit import Chem

from openfe.setup import LigandAtomMapping


def test_atommapping_usage(simple_mapping):
    assert simple_mapping.molA_to_molB[1] == 1
    assert simple_mapping.molA_to_molB.get(2, None) is None

    with pytest.raises(KeyError):
        simple_mapping.molA_to_molB[3]


def test_atommapping_hash(simple_mapping, other_mapping):
    # these two mappings map the same molecules, but with a different mapping
    assert simple_mapping is not other_mapping


def test_draw_mapping_cairo(tmpdir, simple_mapping):
    with tmpdir.as_cwd():
        simple_mapping.draw_to_file('test.png')
        filed = pathlib.Path('test.png')
        assert filed.exists()


def test_draw_mapping_svg(tmpdir, other_mapping):
    with tmpdir.as_cwd():
        d2d = Chem.Draw.rdMolDraw2D.MolDraw2DSVG(600, 300, 300, 300)
        other_mapping.draw_to_file('test.svg', d2d=d2d)
        filed = pathlib.Path('test.svg')
        assert filed.exists()


class TestLigandAtomMappingSerialization:
    def test_to_dict(self, benzene_phenol_mapping):
        d = benzene_phenol_mapping.to_dict()

        assert isinstance(d, dict)
        assert 'molA' in d
        assert 'molB' in d
        assert isinstance(d['molA'], str)

    def test_deserialize_roundtrip(self, benzene_phenol_mapping,
                                   benzene_anisole_mapping):

        roundtrip = LigandAtomMapping.from_dict(
                        benzene_phenol_mapping.to_dict())

        assert roundtrip == benzene_phenol_mapping

        # We don't check coordinates since that's already done in guefe for
        # SmallMoleculeComponent

        assert roundtrip != benzene_anisole_mapping

    def test_file_roundtrip(self, benzene_phenol_mapping, tmpdir):
        with tmpdir.as_cwd():
            with open('tmpfile.json', 'w') as f:
                f.write(benzene_phenol_mapping.to_json())

            with open('tmpfile.json', 'r') as f:
                d = json.load(f)

            assert isinstance(d, dict)
            roundtrip = LigandAtomMapping.from_dict(d)

            assert roundtrip == benzene_phenol_mapping
