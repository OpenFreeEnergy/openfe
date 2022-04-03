import pytest
import pathlib
from rdkit import Chem


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
