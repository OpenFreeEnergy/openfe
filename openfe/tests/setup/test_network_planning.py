# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from rdkit import Chem
import pytest
import networkx as nx

import openfe.setup

from ..conftest import mol_from_smiles


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


@pytest.mark.parametrize('as_list', [False, True])
def test_radial_network(atom_mapping_basic_test_files, toluene_vs_others,
                        as_list):
    toluene, others = toluene_vs_others
    central_ligand_name = 'toluene'
    mapper = openfe.setup.atom_mapping.LomapAtomMapper()
    if as_list:
        mapper = [mapper]

    network = openfe.setup.ligand_network_planning.generate_radial_network(
        ligands=others, central_ligand=toluene,
        mappers=mapper, scorer=None,
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


@pytest.mark.parametrize('with_progress', [True, False])
@pytest.mark.parametrize('with_scorer', [True, False])
@pytest.mark.parametrize('extra_mapper', [True, False])
def test_generate_maximal_network(toluene_vs_others, with_progress,
                                  with_scorer, extra_mapper):
    toluene, others = toluene_vs_others
    if extra_mapper:
        mappers = [
            openfe.setup.atom_mapping.LomapAtomMapper(),
            BadMapper()
        ]
    else:
        mappers = openfe.setup.atom_mapping.LomapAtomMapper()

    def scoring_func(mapping):
        return 1.0 / len(mapping.componentA_to_componentB)

    scorer = scoring_func if with_scorer else None

    network = openfe.setup.ligand_network_planning.generate_maximal_network(
        ligands=others + [toluene],
        mappers=mappers,
        scorer=scorer,
        progress=with_progress,
    )

    assert len(network.nodes) == len(others) + 1

    if extra_mapper:
        edge_count = len(others) * (len(others) + 1)
    else:
        edge_count = len(others) * (len(others) + 1) / 2

    assert len(network.edges) == edge_count

    if scorer:
        for edge in network.edges:
            score = edge.annotations['score']
            assert score == 1.0 / len(edge.componentA_to_componentB)
    else:
        for edge in network.edges:
            assert 'score' not in edge.annotations


@pytest.mark.parametrize('multi_mappers', [False, True])
def test_minimal_spanning_network_mappers(atom_mapping_basic_test_files, multi_mappers):
    ligands = [atom_mapping_basic_test_files['toluene'],
               atom_mapping_basic_test_files['2-naftanol'],
               ]

    if multi_mappers:
        mappers = [BadMapper(), openfe.setup.atom_mapping.LomapAtomMapper()]
    else:
        mappers = openfe.setup.atom_mapping.LomapAtomMapper()

    def scorer(mapping):
        return 1.0 / len(mapping.componentA_to_componentB)

    network = openfe.ligand_network_planning.generate_minimal_spanning_network(
        ligands=ligands,
        mappers=mappers,
        scorer=scorer,
    )

    assert isinstance(network, openfe.LigandNetwork)
    assert list(network.edges)


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
    ref = sorted([
        ('1,3,7-trimethylnaphthalene', '2,6-dimethylnaphthalene'),
        ('1-butyl-4-methylbenzene', '2-methyl-6-propylnaphthalene'),
        ('2,6-dimethylnaphthalene', '2-methyl-6-propylnaphthalene'),
        ('2,6-dimethylnaphthalene', '2-methylnaphthalene'),
        ('2,6-dimethylnaphthalene', '2-naftanol'),
        ('2,6-dimethylnaphthalene', 'methylcyclohexane'),
        ('2,6-dimethylnaphthalene', 'toluene'),
    ])

    assert len(edge_ids) == len(ref)
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


def test_network_from_names(atom_mapping_basic_test_files):
    ligs = list(atom_mapping_basic_test_files.values())

    requested = [
        ('toluene', '2-naftanol'),
        ('2-methylnaphthalene', '2-naftanol'),
    ]

    network = openfe.setup.ligand_network_planning.generate_network_from_names(
        ligands=ligs,
        names=requested,
        mapper=openfe.LomapAtomMapper(),
    )

    assert len(network.nodes) == len(ligs)
    assert len(network.edges) == 2
    actual_edges = [(e.componentA.name, e.componentB.name)
                    for e in network.edges]
    assert set(requested) == set(actual_edges)


def test_network_from_names_bad_name(atom_mapping_basic_test_files):
    ligs = list(atom_mapping_basic_test_files.values())

    requested = [
        ('hank', '2-naftanol'),
        ('2-methylnaphthalene', '2-naftanol'),
    ]

    with pytest.raises(KeyError, match="Invalid name"):
        _ = openfe.setup.ligand_network_planning.generate_network_from_names(
            ligands=ligs,
            names=requested,
            mapper=openfe.LomapAtomMapper(),
        )


def test_network_from_names_duplicate_name(atom_mapping_basic_test_files):
    ligs = list(atom_mapping_basic_test_files.values())
    ligs = ligs + [ligs[0]]

    requested = [
        ('toluene', '2-naftanol'),
        ('2-methylnaphthalene', '2-naftanol'),
    ]

    with pytest.raises(ValueError, match="Duplicate names"):
        _ = openfe.setup.ligand_network_planning.generate_network_from_names(
            ligands=ligs,
            names=requested,
            mapper=openfe.LomapAtomMapper(),
        )


def test_network_from_indices(atom_mapping_basic_test_files):
    ligs = list(atom_mapping_basic_test_files.values())

    requested = [(0, 1), (2, 3)]

    network = openfe.setup.ligand_network_planning.generate_network_from_indices(
        ligands=ligs,
        indices=requested,
        mapper=openfe.LomapAtomMapper(),
    )

    assert len(network.nodes) == len(ligs)
    assert len(network.edges) == 2

    edges = list(network.edges)
    expected_edges = {(ligs[0], ligs[1]), (ligs[2], ligs[3])}
    actual_edges = {(edges[0].componentA, edges[0].componentB),
                    (edges[1].componentA, edges[1].componentB)}

    assert actual_edges == expected_edges


def test_network_from_indices_indexerror(atom_mapping_basic_test_files):
    ligs = list(atom_mapping_basic_test_files.values())

    requested = [(20, 1), (2, 3)]

    with pytest.raises(IndexError, match="Invalid ligand id"):
        network = openfe.setup.ligand_network_planning.generate_network_from_indices(
            ligands=ligs,
            indices=requested,
            mapper=openfe.LomapAtomMapper(),
        )
