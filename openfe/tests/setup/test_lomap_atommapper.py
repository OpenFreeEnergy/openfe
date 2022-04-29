# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest
from rdkit import Chem


import openfe
from openfe.setup import LomapAtomMapper, SmallMoleculeComponent


def test_simple(lomap_basic_test_files):
    # basic sanity check on the LigandAtomMapper
    mol1 = lomap_basic_test_files['methylcyclohexane']
    mol2 = lomap_basic_test_files['toluene']

    mapper = LomapAtomMapper()

    mapping_gen = mapper.suggest_mappings(mol1, mol2)

    mapping = next(mapping_gen)
    assert isinstance(mapping, openfe.setup.LigandAtomMapping)
    # maps (CH3) off methyl and (6C + 5H) on ring
    assert len(mapping.molA_to_molB) == 15


def test_generator_length(lomap_basic_test_files):
    # check that we get one mapping back from Lomap LigandAtomMapper then the
    # generator stops correctly
    mol1 = lomap_basic_test_files['methylcyclohexane']
    mol2 = lomap_basic_test_files['toluene']

    mapper = LomapAtomMapper()

    mapping_gen = mapper.suggest_mappings(mol1, mol2)

    _ = next(mapping_gen)
    with pytest.raises(StopIteration):
        next(mapping_gen)


def test_bad_mapping(lomap_basic_test_files):
    toluene = lomap_basic_test_files['toluene']
    NigelTheNitrogen = SmallMoleculeComponent(Chem.MolFromSmiles('N'), name='Nigel')

    mapper = LomapAtomMapper()

    mapping_gen = mapper.suggest_mappings(toluene, NigelTheNitrogen)
    with pytest.raises(StopIteration):
        next(mapping_gen)
