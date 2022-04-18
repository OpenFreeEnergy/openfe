# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from rdkit import Chem
import pytest

import openfe.setup


class BadMapper(openfe.setup.LigandAtomMapper):
    def _mappings_generator(self, molA, molB):
        yield {0: 0}


@pytest.fixture
def toluene_vs_others(lomap_basic_test_files):
    central_ligand_name = 'toluene'
    others = [v for (k, v) in lomap_basic_test_files.items()
              if k != central_ligand_name]
    toluene = lomap_basic_test_files[central_ligand_name]
    return toluene, others


def test_radial_graph(lomap_basic_test_files, toluene_vs_others):
    toluene, others = toluene_vs_others
    central_ligand_name = 'toluene'
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
    assert all((central_ligand_name in {mapping.molA.name, mapping.molB.name})
               for mapping in network.edges)


def test_radial_graph_with_scorer(toluene_vs_others):
    toluene, others = toluene_vs_others

    def scorer(mapping):
        return 1.0 / len(mapping.molA_to_molB)

    network = openfe.setup.ligand_network_planning.generate_radial_network(
        ligands=others,
        central_ligand=toluene,
        mappers=[BadMapper(), openfe.setup.LomapAtomMapper()],
        scorer=scorer
    )
    assert len(network.edges) == len(others)

    for edge in network.edges:
        assert len(edge.molA_to_molB) > 1  # we didn't take the bad mapper
        assert 'score' in edge.annotations
        assert edge.annotations['score'] == 1.0 / len(edge.molA_to_molB)


def test_radial_graph_multiple_mappers_no_scorer(toluene_vs_others):
    toluene, others = toluene_vs_others
    mappers = [BadMapper(), openfe.setup.LomapAtomMapper()]
    # in this one, we should always take the bad mapper
    network = openfe.setup.ligand_network_planning.generate_radial_network(
        ligands=others,
        central_ligand=toluene,
        mappers=[BadMapper(), openfe.setup.LomapAtomMapper()]
    )
    assert len(network.edges) == len(others)

    for edge in network.edges:
        assert edge.molA_to_molB == {0: 0}


def test_radial_network_failure(lomap_basic_test_files):
    nigel = openfe.setup.SmallMoleculeComponent(Chem.MolFromSmiles('N'))

    with pytest.raises(ValueError, match='No mapping found for'):
        network = openfe.setup.ligand_network_planning.generate_radial_network(
            ligands=[nigel], central_ligand=lomap_basic_test_files['toluene'],
            mappers=[openfe.setup.LomapAtomMapper()], scorer=None
        )
