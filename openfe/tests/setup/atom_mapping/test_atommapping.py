import openfe.setup
import pytest
import pathlib
import json
from rdkit import Chem

from openfe.setup.atom_mapping import LigandAtomMapping


@pytest.fixture
def annotated_simple_mapping(simple_mapping):
    mapping = LigandAtomMapping(simple_mapping.molA,
                                simple_mapping.molB,
                                simple_mapping.molA_to_molB,
                                annotations={'foo': 'bar'})
    return mapping


def test_atommapping_usage(simple_mapping):
    assert simple_mapping.molA_to_molB[1] == 1
    assert simple_mapping.molA_to_molB.get(2, None) is None
    assert simple_mapping.annotations == {}

    with pytest.raises(KeyError):
        simple_mapping.molA_to_molB[3]

def test_mapping_inversion(benzene_phenol_mapping):
    assert benzene_phenol_mapping.molB_to_molA == {0: 0, 1: 1, 2: 2, 3: 3,
                                                   4: 4, 5: 5, 6: 6, 7: 7,
                                                   8: 8, 9: 9, 11: 11, 12: 10}

def test_uniques(atom_mapping_basic_test_files):
    mapping = openfe.setup.LigandAtomMapping(
        molA=atom_mapping_basic_test_files['methylcyclohexane'],
        molB=atom_mapping_basic_test_files['toluene'],
        molA_to_molB={
            0: 6, 1: 7, 2: 8, 3: 9, 4: 10, 5: 11, 6: 12
        }
    )

    assert list(mapping.componentA_unique) == [7, 8, 9, 10, 11, 12, 13, 14, 15,
                                               16, 17, 18, 19, 20]
    assert list(mapping.componentB_unique) == [0, 1, 2, 3, 4, 5, 13, 14]


def test_modification(benzene_phenol_mapping):
    # check that we get a copy of the mapping and we can't modify
    AtoB = benzene_phenol_mapping.molA_to_molB
    before = len(AtoB)

    AtoB.pop(10)

    assert len(benzene_phenol_mapping.molA_to_molB) == before

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
        assert 'annotations' in d
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
                f.write(json.dumps(benzene_phenol_mapping.to_dict()))

            with open('tmpfile.json', 'r') as f:
                d = json.load(f)

            assert isinstance(d, dict)
            roundtrip = LigandAtomMapping.from_dict(d)

            assert roundtrip == benzene_phenol_mapping


def test_annotated_atommapping_hash_eq(simple_mapping,
                                       annotated_simple_mapping):
    assert annotated_simple_mapping != simple_mapping
    assert hash(annotated_simple_mapping) != hash(simple_mapping)


def test_annotation_immutability(annotated_simple_mapping):
    annot1 = annotated_simple_mapping.annotations
    annot1['foo'] = 'baz'
    annot2 = annotated_simple_mapping.annotations
    assert annot1 != annot2
    assert annot2 == {'foo': 'bar'}


def test_with_annotations(simple_mapping, annotated_simple_mapping):
    new_annot = simple_mapping.with_annotations({'foo': 'bar'})
    assert new_annot == annotated_simple_mapping
