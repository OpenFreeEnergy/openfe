# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing import Dict, Tuple

from rdkit import Chem
from openfe.setup import SmallMoleculeComponent
import lomap
import pytest

from openfe.setup import LigandAtomMapping

from ..conftest import mol_from_smiles


def _translate_lomap_mapping(atom_mapping_str: str) -> Dict[int, int]:
    mapped_atom_tuples = map(lambda x: tuple(map(int, x.split(":"))),
                             atom_mapping_str.split(","))
    return {i: j for i, j in mapped_atom_tuples}


def _get_atom_mapping_dict(lomap_atom_mappings) -> Dict[Tuple[int, int],
                                                        Dict[int, int]]:
    return {mol_pair: _translate_lomap_mapping(atom_mapping_str) for
            mol_pair, atom_mapping_str in
            lomap_atom_mappings.mcs_map_store.items()}


@pytest.fixture()
def gufe_atom_mapping_matrix(lomap_basic_test_files_dir,
                             atom_mapping_basic_test_files
                             ) -> Dict[Tuple[int, int], LigandAtomMapping]:
    dbmols = lomap.DBMolecules(lomap_basic_test_files_dir, verbose='off')
    _, _ = dbmols.build_matrices()
    molecule_pair_atom_mappings = _get_atom_mapping_dict(dbmols)

    ligand_atom_mappings = {}
    for (i, j), val in molecule_pair_atom_mappings.items():
        nm1 = dbmols[i].getName()[:-5]
        nm2 = dbmols[j].getName()[:-5]
        ligand_atom_mappings[(i, j)] = LigandAtomMapping(
            molA=atom_mapping_basic_test_files[nm1],
            molB=atom_mapping_basic_test_files[nm2],
            molA_to_molB=val)

    return ligand_atom_mappings


@pytest.fixture()
def mol_pair_to_shock_perses_mapper() -> Tuple[SmallMoleculeComponent,
                                               SmallMoleculeComponent]:
    """
    This pair of Molecules leads to an empty Atom mapping in
    Perses Mapper with certain settings.

    Returns:
        Tuple[SmallMoleculeComponent]: two molecule objs for the test
    """
    molA = SmallMoleculeComponent(mol_from_smiles('c1ccccc1'), 'benzene')
    molB = SmallMoleculeComponent(mol_from_smiles('C1CCCCC1'), 'cyclohexane')
    return molA, molB
