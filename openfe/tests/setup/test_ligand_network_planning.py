# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from rdkit import Chem
import pytest
import networkx as nx

import openfe.setup

from .conftest import mol_from_smiles


class BadMapper(openfe.setup.atom_mapping.LigandAtomMapper):
    def _mappings_generator(self, molA, molB):
        yield {0: 0}


@pytest.fixture(scope='session')
def toluene_vs_others(atom_mapping_basic_test_files):
    central_ligand_name = 'toluene'
    others = [v for (k, v) in atom_mapping_basic_test_files.items()
              if k != central_ligand_name]
    toluene = atom_mapping_basic_test_files[central_ligand_name]
    return toluene, others


def test_radial_network(atom_mapping_basic_test_files, toluene_vs_others):
    toluene, others = toluene_vs_others
    central_ligand_name = 'toluene'
    mapper = openfe.setup.atom_mapping.LomapAtomMapper()

    network = openfe.setup.ligand_network_planning.generate_radial_network(
        ligands=others, central_ligand=toluene,
        mappers=[mapper], scorer=None,
    )
    # couple sanity checks
    assert len(network.nodes) == len(atom_mapping_basic_test_files)
    assert len(network.edges) == len(others)
    # check that all ligands are present, i.e. we included everyone
    ligands_in_network = {mol.name for mol in network.nodes}
    assert ligands_in_network == set(atom_mapping_basic_test_files.keys())
    # check that every edge has the central ligand within
    assert all((central_ligand_name in {mapping.componentA.name, mapping.componentB.name})
               for mapping in network.edges)


def test_radial_network_with_scorer(toluene_vs_others):
    toluene, others = toluene_vs_others

    def scorer(mapping):
        return 1.0 / len(mapping.componentA_to_componentB)

    network = openfe.setup.ligand_network_planning.generate_radial_network(
        ligands=others,
        central_ligand=toluene,
        mappers=[BadMapper(), openfe.setup.atom_mapping.LomapAtomMapper()],
        scorer=scorer
    )
    assert len(network.edges) == len(others)

    for edge in network.edges:
        assert len(edge.componentA_to_componentB) > 1  # we didn't take the bad mapper
        assert 'score' in edge.annotations
        assert edge.annotations['score'] == 1.0 / len(edge.componentA_to_componentB)


def test_radial_network_multiple_mappers_no_scorer(toluene_vs_others):
    toluene, others = toluene_vs_others
    mappers = [BadMapper(), openfe.setup.atom_mapping.LomapAtomMapper()]
    # in this one, we should always take the bad mapper
    network = openfe.setup.ligand_network_planning.generate_radial_network(
        ligands=others,
        central_ligand=toluene,
        mappers=[BadMapper(), openfe.setup.atom_mapping.LomapAtomMapper()]
    )
    assert len(network.edges) == len(others)

    for edge in network.edges:
        assert edge.componentA_to_componentB == {0: 0}


def test_radial_network_failure(atom_mapping_basic_test_files):
    nigel = openfe.SmallMoleculeComponent(mol_from_smiles('N'))

    with pytest.raises(ValueError, match='No mapping found for'):
        network = openfe.setup.ligand_network_planning.generate_radial_network(
            ligands=[nigel],
            central_ligand=atom_mapping_basic_test_files['toluene'],
            mappers=[openfe.setup.atom_mapping.LomapAtomMapper()],
            scorer=None
        )


@pytest.fixture(scope='session')
def minimal_spanning_network(toluene_vs_others):
    toluene, others = toluene_vs_others
    mappers = [BadMapper(), openfe.setup.atom_mapping.LomapAtomMapper()]

    def scorer(mapping):
        return 1.0 / len(mapping.componentA_to_componentB)

    network = openfe.setup.ligand_network_planning.generate_minimal_spanning_network(
        ligands=others + [toluene],
        mappers=mappers,
        scorer=scorer
    )
    return network


def test_minimal_spanning_network(minimal_spanning_network, toluene_vs_others):
    tol, others = toluene_vs_others
    assert len(minimal_spanning_network.nodes) == len(others) + 1
    for edge in minimal_spanning_network.edges:
        assert edge.componentA_to_componentB != {0: 0}  # lomap should find something


def test_minimal_spanning_network_connectedness(minimal_spanning_network):
    found_pairs = set()
    for edge in minimal_spanning_network.edges:
        pair = frozenset([edge.componentA, edge.componentB])
        assert pair not in found_pairs
        found_pairs.add(pair)

    assert nx.is_connected(nx.MultiGraph(minimal_spanning_network.graph))


def test_minimal_spanning_network_regression(minimal_spanning_network):
    # issue #244, this was previously giving non-reproducible (yet valid)
    # networks when scores were tied.
    edge_ids = sorted(
        (edge.componentA.name, edge.componentB.name)
        for edge in minimal_spanning_network.edges
    )
    ref = sorted([('1,3,7-trimethylnaphthalene', '2-naftanol'),
           ('1,3,7-trimethylnaphthalene', 'methylcyclohexane'),
           ('1,3,7-trimethylnaphthalene', '2-methylnaphthalene'),
           ('1,3,7-trimethylnaphthalene', '2,6-dimethylnaphthalene'),
           ('1,3,7-trimethylnaphthalene', 'toluene'),
           ('1,3,7-trimethylnaphthalene', '2-methyl-6-propylnaphthalene'),
           ('1-butyl-4-methylbenzene', '2-methyl-6-propylnaphthalene')])

    assert edge_ids == ref


def test_minimal_spanning_network_unreachable(toluene_vs_others):
    toluene, others = toluene_vs_others
    nimrod = openfe.SmallMoleculeComponent(mol_from_smiles("N"))

    def scorer(mapping):
        return 1.0 / len(mapping.componentA_to_componentB)

    with pytest.raises(RuntimeError, match="Unable to create edges"):
        network = openfe.setup.ligand_network_planning.generate_minimal_spanning_network(
            ligands=others + [toluene, nimrod],
            mappers=[openfe.setup.atom_mapping.LomapAtomMapper()],
            scorer=scorer
        )
