# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import openfe
from gufe import SmallMoleculeComponent
from openfe.setup.atom_mapping import LomapAtomMapper

from .conftest import mol_from_smiles


def test_simple(atom_mapping_basic_test_files):
    # basic sanity check on the LigandAtomMapper
    mol1 = atom_mapping_basic_test_files['methylcyclohexane']
    mol2 = atom_mapping_basic_test_files['toluene']

    mapper = LomapAtomMapper()

    mapping_gen = mapper.suggest_mappings(mol1, mol2)

    mapping = next(mapping_gen)
    assert isinstance(mapping, openfe.setup.atom_mapping.LigandAtomMapping)
    # maps (CH3) off methyl and (6C + 5H) on ring
    assert len(mapping.componentA_to_componentB) == 15


def test_distances(atom_mapping_basic_test_files):
    # basic sanity check on the LigandAtomMapper
    mol1 = atom_mapping_basic_test_files['methylcyclohexane']
    mol2 = atom_mapping_basic_test_files['toluene']

    mapper = LomapAtomMapper()
    mapping = next(mapper.suggest_mappings(mol1, mol2))

    dists = mapping.get_distances()

    assert len(dists) == len(mapping.componentA_to_componentB)
    i, j = next(iter(mapping.componentA_to_componentB.items()))
    ref_d = mol1.to_rdkit().GetConformer().GetAtomPosition(i).Distance(
        mol2.to_rdkit().GetConformer().GetAtomPosition(j)
    )
    assert pytest.approx(dists[0], rel=1e-5) == ref_d
    assert pytest.approx(dists[0], rel=1e-5) == 0.07249779


def test_generator_length(atom_mapping_basic_test_files):
    # check that we get one mapping back from Lomap LigandAtomMapper then the
    # generator stops correctly
    mol1 = atom_mapping_basic_test_files['methylcyclohexane']
    mol2 = atom_mapping_basic_test_files['toluene']

    mapper = LomapAtomMapper()

    mapping_gen = mapper.suggest_mappings(mol1, mol2)

    _ = next(mapping_gen)
    with pytest.raises(StopIteration):
        next(mapping_gen)


def test_bad_mapping(atom_mapping_basic_test_files):
    toluene = atom_mapping_basic_test_files['toluene']
    NigelTheNitrogen = SmallMoleculeComponent(mol_from_smiles('N'),
                                              name='Nigel')

    mapper = LomapAtomMapper()

    mapping_gen = mapper.suggest_mappings(toluene, NigelTheNitrogen)
    with pytest.raises(StopIteration):
        next(mapping_gen)


# TODO: Remvoe these test when element changes are allowed - START
def test_simple_no_element_changes(atom_mapping_basic_test_files):
    # basic sanity check on the LigandAtomMapper
    mol1 = atom_mapping_basic_test_files['methylcyclohexane']
    mol2 = atom_mapping_basic_test_files['toluene']

    mapper = LomapAtomMapper()
    mapper._no_element_changes = True
    mapping_gen = mapper.suggest_mappings(mol1, mol2)

    mapping = next(mapping_gen)
    assert isinstance(mapping, openfe.setup.atom_mapping.LigandAtomMapping)
    # maps (CH3) off methyl and (6C + 5H) on ring
    assert len(mapping.componentA_to_componentB) == 15
    
def test_simple_no_element_changes_err(atom_mapping_basic_test_files):
    rdmol1 = Chem.MolFromSmiles("NO")
    rdmol2 = Chem.MolFromSmiles("CC")
    
    Chem.rdDistGeom.EmbedMolecule(rdmol1)
    Chem.rdDistGeom.EmbedMolecule(rdmol2)

    mapper = LomapAtomMapper()
    mapper._no_element_changes = True
    mapping_gen = mapper.suggest_mappings(SmallMoleculeComponent(rdmol1), SmallMoleculeComponent(rdmol2))

    with pytest.raises(ValueError, match="Could not map ligands - Element Changes are not allowed currently."):
        mapping = next(mapping_gen)

    
def test_bas_mapping_no_element_changes(atom_mapping_basic_test_files):
    toluene = atom_mapping_basic_test_files['toluene']
    NigelTheNitrogen = SmallMoleculeComponent(mol_from_smiles('N'),
                                              name='Nigel')

    mapper = LomapAtomMapper()
    mapper._no_element_changes = True
    mapping_gen = mapper.suggest_mappings(toluene, NigelTheNitrogen)
    with pytest.raises(StopIteration):
        next(mapping_gen)
        
# TODO: Remvoe these test when element changes are allowed - END

