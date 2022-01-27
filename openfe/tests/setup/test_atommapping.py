import pytest

from openfe.setup import AtomMapping


@pytest.fixture
def simple_mapping():
    """
    A0-A1-A2-A3

    to

    B0-B1----B2
    """
    m = AtomMapping()
    m.AtoB[0] = 0
    m.AtoB[1] = 1
    m.AtoB[2] = None
    m.AtoB[3] = 2

    m.BtoA[0] = 0
    m.BtoA[1] = 1
    m.BtoA[2] = 2

    return m


def test_atommapping_usage(simple_mapping):
    assert len(simple_mapping.AtoB) == 4
    assert len(simple_mapping.BtoA) == 3

    assert simple_mapping.AtoB[2] is None
    assert simple_mapping.AtoB[3] == 2

    assert simple_mapping.BtoA[2] == 2
    with pytest.raises(KeyError):
        simple_mapping.AtoB[4]
    with pytest.raises(KeyError):
        simple_mapping.BtoA[3]
