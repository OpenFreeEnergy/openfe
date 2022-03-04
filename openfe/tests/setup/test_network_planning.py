# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from rdkit import Chem
import pytest

import openfe.setup


def test_radial_graph(lomap_basic_test_files):
    central_ligand_name = 'toluene'
    others = [v for (k, v) in lomap_basic_test_files.items()
              if k != central_ligand_name]
    toluene = lomap_basic_test_files[central_ligand_name]
    mapper = openfe.setup.LomapAtomMapper()

    network = openfe.setup.ligand_network_planning.generate_radial_network(
        ligands=others, central_ligand=toluene,
        mappers=[mapper], scorer=None,
    )
    # couple sanity checks
    assert len(network.nodes) == len(lomap_basic_test_files)
    assert len(network.edges) == len(others)
    # check that all ligands are present, i.e. we included everyone
    ligands_in_network = {mol.name for mol in network.nodes}
    assert ligands_in_network == set(lomap_basic_test_files.keys())
    # check that every edge has the central ligand within
    assert all((central_ligand_name in {mapping.mol1.name, mapping.mol2.name})
               for mapping in network.edges)


def test_radial_network_failure(lomap_basic_test_files):
    nigel = openfe.setup.LigandMolecule(Chem.MolFromSmiles('N'))

    with pytest.raises(ValueError, match='No mapping found for'):
        network = openfe.setup.ligand_network_planning.generate_radial_network(
            ligands=[nigel], central_ligand=lomap_basic_test_files['toluene'],
            mappers=[openfe.setup.LomapAtomMapper()], scorer=None
        )
