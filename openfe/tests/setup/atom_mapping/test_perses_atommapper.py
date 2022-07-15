# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest
from rdkit import Chem

import openfe
from openfe.setup import SmallMoleculeComponent
from openfe.setup.atom_mapping import PersesAtomMapper, LigandAtomMapping

pytest.importorskip('perses')
pytest.importorskip('openeye')


def test_simple(atom_mapping_basic_test_files):
    # basic sanity check on the LigandAtomMapper
    mol1 = atom_mapping_basic_test_files['methylcyclohexane']
    mol2 = atom_mapping_basic_test_files['toluene']

    mapper = PersesAtomMapper()

    mapping_gen = mapper.suggest_mappings(mol1, mol2)

    mapping = next(mapping_gen)
    assert isinstance(mapping, LigandAtomMapping)
    # maps (CH3) off methyl and (6C + 5H) on ring
    assert len(mapping.molA_to_molB) == 4


def test_generator_length(atom_mapping_basic_test_files):
    # check that we get one mapping back from Lomap LigandAtomMapper then the
    # generator stops correctly
    mol1 = atom_mapping_basic_test_files['methylcyclohexane']
    mol2 = atom_mapping_basic_test_files['toluene']

    mapper = PersesAtomMapper()

    mapping_gen = mapper.suggest_mappings(mol1, mol2)

    _ = next(mapping_gen)
    with pytest.raises(StopIteration):
        next(mapping_gen)

def test_empty_atommappings(mol_pair_to_shock_perses_mapper):
    mol1, mol2 = mol_pair_to_shock_perses_mapper
    mapper = PersesAtomMapper()

    mapping_gen = mapper.suggest_mappings(mol1, mol2)
    
    #The expected return is an empty mapping
    assert len(list(mapping_gen)) == 0
    
    with pytest.raises(StopIteration):
        next(mapping_gen)
    