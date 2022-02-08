import pytest


from openfe.setup.lomap_mapper import LomapAtomMapper


def test_simple(lomap_basic_test_files):
    mol1 = lomap_basic_test_files['methylcyclohexane']
    mol2 = lomap_basic_test_files['toluene']

    mapper = LomapAtomMapper()

    mapping_gen = mapper.suggest_mappings(mol1, mol2)

    mapping = next(mapping_gen)

    assert len(mapping.mol1_to_mol2) == 7

    with pytest.raises(StopIteration):
        next(mapping_gen)
