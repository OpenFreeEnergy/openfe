import pytest


def test_atommapping_usage(simple_mapping):
    assert simple_mapping.mol1_to_mol2[1] == 1
    assert simple_mapping.mol1_to_mol2.get(2, None) is None

    with pytest.raises(KeyError):
        simple_mapping.mol1_to_mol2[3]


def test_atommapping_hash(simple_mapping, other_mapping):
    # these two mappings map the same molecules, but with a different mapping
    assert simple_mapping is not other_mapping


def test_to_explicit(simple_mapping):
    explicit_simple_mapping = simple_mapping.to_explicit()

    # CH3-CH3 -> CH3-CH2-OH
    # both carbons map, and 5 hydrogens, so 7?
    assert len(explicit_simple_mapping.mol1_to_mol2) == 7
