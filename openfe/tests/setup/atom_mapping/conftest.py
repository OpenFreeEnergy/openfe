# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing import Dict, Tuple

import lomap

import pytest

from openfe.setup import LigandAtomMapping


def _translate_lomap_mapping(atom_mapping_str: str) -> Dict[int, int]:
    return dict(map(lambda x: tuple(map(int, x.split(":"))),
                    atom_mapping_str.split(",")))


def _get_atom_mapping_dict(lomap_atom_mappings)->Dict[Tuple[int, int],
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
