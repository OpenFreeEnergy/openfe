import pytest
from rdkit import Chem


import openfe
from openfe.setup import Molecule
from openfe.setup.lomap_mapper import LomapAtomMapper


def test_simple(lomap_basic_test_files):
    # basic sanity check on the AtomMapper
    mol1 = Molecule(lomap_basic_test_files['methylcyclohexane'])
    mol2 = Molecule(lomap_basic_test_files['toluene'])

    mapper = LomapAtomMapper()

    mapping_gen = mapper.suggest_mappings(mol1, mol2)

    mapping = next(mapping_gen)
    assert isinstance(mapping, openfe.setup.AtomMapping)
    # methylcyclohexane to toluene is a 1:1 mapping between all atoms
    # so 7 values should be present
    assert len(mapping.mol1_to_mol2) == 7


def test_generator_length(lomap_basic_test_files):
    # check that we get one mapping back from Lomap AtomMapper then the
    # generator stops correctly
    mol1 = Molecule(lomap_basic_test_files['methylcyclohexane'])
    mol2 = Molecule(lomap_basic_test_files['toluene'])

    mapper = LomapAtomMapper()

    mapping_gen = mapper.suggest_mappings(mol1, mol2)

    _ = next(mapping_gen)
    with pytest.raises(StopIteration):
        next(mapping_gen)


def test_bad_mapping(lomap_basic_test_files):
    toluene = Molecule(lomap_basic_test_files['toluene'])
    NigelTheNitrogen = Molecule(Chem.MolFromSmiles('N'))

    mapper = LomapAtomMapper()

    mapping_gen = mapper.suggest_mappings(toluene, NigelTheNitrogen)
    with pytest.raises(StopIteration):
        next(mapping_gen)
