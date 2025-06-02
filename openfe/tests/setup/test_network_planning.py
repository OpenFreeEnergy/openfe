# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing import Callable
import pytest

import openfe

from ..conftest import mol_from_smiles


class BadMapper(openfe.setup.atom_mapping.LigandAtomMapper):
    @classmethod
    def _defaults(cls):
        return {}

    def _to_dict(self):
        return {}

    @classmethod
    def _from_dict(cls, d):
        return cls()

    def _mappings_generator(self, molA, molB):
        yield {0: 0}


@pytest.fixture()
def toluene_vs_others(atom_mapping_basic_test_files):
    central_ligand_name = 'toluene'
    others = [v for (k, v) in atom_mapping_basic_test_files.items()
              if k != central_ligand_name]
    toluene = atom_mapping_basic_test_files[central_ligand_name]
    return toluene, others


@pytest.fixture()
def simple_scorer() -> Callable:
    def _scorer(mapping) -> float:
        "Returns a score proportional to the length of the mapping, normalized to be in [0,1]"
        return 1 - (1 / len(mapping.componentA_to_componentB))
    return _scorer

@pytest.fixture()
def deterministic_toluene_mst_scorer() -> Callable:
    def _scorer(mapping)-> float:
        """These scores give the same mst or rmst every time for the toluene_vs_others dataset."""
        scores = {
            # MST edges
            ('1,3,7-trimethylnaphthalene', '2,6-dimethylnaphthalene'): 0.3,
            ('1-butyl-4-methylbenzene', '2-methyl-6-propylnaphthalene'): 0.3,
            ('2,6-dimethylnaphthalene', '2-methyl-6-propylnaphthalene'): 0.3,
            ('2,6-dimethylnaphthalene', '2-methylnaphthalene'): 0.3,
            ('2,6-dimethylnaphthalene', '2-naftanol'): 0.3,
            ('2,6-dimethylnaphthalene', 'methylcyclohexane'): 0.3,
            ('2,6-dimethylnaphthalene', 'toluene'): 0.3,
            # MST redundant edges
            ('1,3,7-trimethylnaphthalene', '2-methyl-6-propylnaphthalene'): 0.2,
            ('1-butyl-4-methylbenzene', '2,6-dimethylnaphthalene'): 0.2,
            ('1-butyl-4-methylbenzene', 'toluene'): 0.2,
            ('2-methyl-6-propylnaphthalene', '2-methylnaphthalene'): 0.2,
            ('2-methylnaphthalene', '2-naftanol'): 0.2,
            ('2-methylnaphthalene', 'methylcyclohexane'): 0.2,
            ('2-methylnaphthalene', 'toluene'): 0.2,
        }
        return scores.get((mapping.componentA.name, mapping.componentB.name), 0.1)
    return _scorer


@pytest.fixture()
def deterministic_minimal_spanning_network(toluene_vs_others, lomap_old_mapper, deterministic_toluene_mst_scorer):
    # TODO: I'm not convinced this needs to be its own fixture
    toluene, others = toluene_vs_others
    scorer = deterministic_toluene_mst_scorer

    network = openfe.setup.ligand_network_planning.generate_minimal_spanning_network(
        ligands=others + [toluene],
        mappers=lomap_old_mapper,
        scorer=scorer,
    )
    return network


@pytest.fixture()
def deterministic_minimal_redundant_network(toluene_vs_others, lomap_old_mapper, deterministic_toluene_mst_scorer):
    # TODO: I'm not convinced this needs to be its own fixture
    toluene, others = toluene_vs_others
    scorer = deterministic_toluene_mst_scorer

    network = openfe.setup.ligand_network_planning.generate_minimal_redundant_network(
        ligands=others + [toluene],
        mappers=lomap_old_mapper,
        scorer=scorer,
        mst_num=2
    )
    return network

class TestRadialNetworkGenerator:
    @pytest.mark.parametrize("as_list", [False, True])
    def test_radial_network(
        self,
        toluene_vs_others,
        as_list,
        lomap_old_mapper,
    ):
        toluene, others = toluene_vs_others
        central_ligand_name = "toluene"

        mapper = lomap_old_mapper
        if as_list:
            mapper = [mapper]

        network = openfe.setup.ligand_network_planning.generate_radial_network(
            ligands=others,
            central_ligand=toluene,
            mappers=mapper,
            scorer=None,
        )

        expected_names = set([c.name for c in others] + [central_ligand_name])

        # couple sanity checks
        assert len(network.nodes) == len(expected_names)
        assert len(network.edges) == len(expected_names) - 1

        # check that all ligands are present, i.e. we included everyone
        ligands_in_network = {mol.name for mol in network.nodes}
        assert ligands_in_network == expected_names

        # check that every edge contains the central ligand as a node
        assert all(
            (central_ligand_name in {mapping.componentA.name, mapping.componentB.name})
            for mapping in network.edges
        )

    @pytest.mark.parametrize("central_ligand_arg", [0, "toluene"])
    def test_radial_network_central_ligand_int_str(
        self,
        toluene_vs_others,
        central_ligand_arg,
        lomap_old_mapper,
    ):
        """check that passing either an integer or string to indicate the central ligand works"""
        toluene, others = toluene_vs_others
        ligands = [toluene] + others

        network = openfe.setup.ligand_network_planning.generate_radial_network(
            ligands=ligands,
            central_ligand=central_ligand_arg,
            mappers=lomap_old_mapper,
            scorer=None,
        )

        central_ligand_name = "toluene"
        expected_names = set([c.name for c in others] + [central_ligand_name])

        # couple sanity checks
        assert len(network.nodes) == len(expected_names)
        assert len(network.edges) == len(expected_names) - 1

        # check that all ligands are present, i.e. we included everyone
        ligands_in_network = {mol.name for mol in network.nodes}
        assert ligands_in_network == expected_names

        # check that every edge contains the central ligand as a node
        assert all(
            (central_ligand_name in {mapping.componentA.name, mapping.componentB.name})
            for mapping in network.edges
        )

    def test_radial_network_bad_name(self, toluene_vs_others, lomap_old_mapper):
        """Error if the central ligand requested is not present."""
        toluene, others = toluene_vs_others
        ligands = [toluene] + others

        with pytest.raises(ValueError, match="No ligand called 'unobtainium"):
            _ = openfe.setup.ligand_network_planning.generate_radial_network(
                ligands=ligands,
                central_ligand="unobtainium",
                mappers=lomap_old_mapper,
                scorer=None,
            )

    def test_radial_network_multiple_str(self, toluene_vs_others, lomap_old_mapper):
        """Error if more than one ligand has the name passed to 'central_ligand'."""
        toluene, others = toluene_vs_others
        ligands = [toluene, toluene] + others

        with pytest.raises(ValueError, match="Multiple ligands called"):
            _ = openfe.setup.ligand_network_planning.generate_radial_network(
                ligands=ligands,
                central_ligand="toluene",
                mappers=lomap_old_mapper,
                scorer=None,
            )

    def test_radial_network_index_error(self, toluene_vs_others, lomap_old_mapper):
        """Throw a helpful error if the index value passed to 'central_ligand' is out-of-bounds."""
        toluene, others = toluene_vs_others
        ligands = [toluene] + others

        with pytest.raises(ValueError, match="index '2077' out of bounds, there are 8 ligands"):
            openfe.setup.ligand_network_planning.generate_radial_network(
                ligands=ligands,
                central_ligand=2077,
                mappers=lomap_old_mapper,
                scorer=None,
            )

    def test_radial_network_self_central(self, toluene_vs_others, lomap_old_mapper):
        """(issue #544) If the central ligand is included in "ligands",
        there shouldn't be a self-edge to the central ligand, and a warning should be raised.
        """
        toluene, others = toluene_vs_others
        ligands = [toluene] + others

        with pytest.warns(UserWarning, match="The central_ligand toluene was also found in the list of ligands"):
            network = openfe.setup.ligand_network_planning.generate_radial_network(
                ligands=ligands,
                central_ligand=toluene,
                mappers=lomap_old_mapper,
                scorer=None,
            )

        # make sure there's no self-edge for the central ligand (toluene)
        assert ('toluene', 'toluene') not in {(e.componentA.name, e.componentB.name) for e in network.edges}
        assert len(network.edges) == len(ligands) - 1

        # explicitly check to make sure there is no toluene self-edge
        name_pairs =  [(c.componentA, c.componentB) for c in network.edges]
        assert ('toluene', 'toluene') not in name_pairs

    def test_radial_network_with_scorer(self, toluene_vs_others, lomap_old_mapper, simple_scorer):
        """Test that the scorer chooses the mapper with the best score (in this case, the LOMAP mapper)."""
        toluene, others = toluene_vs_others
        mappers = [BadMapper(), lomap_old_mapper]
        scorer = simple_scorer

        network = openfe.setup.ligand_network_planning.generate_radial_network(
            ligands=others,
            central_ligand=toluene,
            mappers=mappers,
            scorer=scorer
        )

        expected_names = set([c.name for c in others] + ['toluene'])

        # couple sanity checks
        assert len(network.nodes) == len(expected_names)
        assert len(network.edges) == len(expected_names) - 1

        # check that all ligands are present, i.e. we included everyone
        ligands_in_network = {mol.name for mol in network.nodes}
        assert ligands_in_network == expected_names

        for edge in network.edges:
            # make sure we didn't take the bad mapper, which would always be a length of 1 ({0:0})
            assert len(edge.componentA_to_componentB) > 1
            assert 'score' in edge.annotations
            assert edge.annotations['score'] == 1 - 1 / len(edge.componentA_to_componentB)

    def test_radial_network_multiple_mappers_no_scorer(self, toluene_vs_others, lomap_old_mapper):
        toluene, others = toluene_vs_others
        mappers = [BadMapper(), lomap_old_mapper]

        network = openfe.setup.ligand_network_planning.generate_radial_network(
            ligands=others,
            central_ligand=toluene,
            mappers=mappers,
        )

        expected_names = set([c.name for c in others] + ['toluene'])

        # couple sanity checks
        assert len(network.nodes) == len(expected_names)
        assert len(network.edges) == len(expected_names) - 1

        # check that all ligands are present, i.e. we included everyone
        ligands_in_network = {mol.name for mol in network.nodes}
        assert ligands_in_network == expected_names

        for edge in network.edges:
            # we should always take the first valid mapper (BadMapper) when there is no scorer.
            assert edge.componentA_to_componentB == {0: 0}
            assert "score" not in edge.annotations

    def test_radial_network_no_mapping_failure(self, toluene_vs_others, lomap_old_mapper):
        """Error if any node does not have a mapping to the central component."""
        toluene, others = toluene_vs_others
        # lomap cannot make a mapping to nimrod, and will return nothing for the (toluene, nimrod) pair
        nimrod = openfe.SmallMoleculeComponent(mol_from_smiles('N'), name='nimrod')

        with pytest.raises(ValueError, match=r'No mapping found for SmallMoleculeComponent\(name=nimrod\)'):
            _ = openfe.setup.ligand_network_planning.generate_radial_network(
                ligands=others + [nimrod],
                central_ligand=toluene,
                mappers=[lomap_old_mapper],
                scorer=None
            )


@pytest.mark.parametrize("with_progress", [True, False])
@pytest.mark.parametrize("with_scorer", [True, False])
@pytest.mark.parametrize("extra_mapper", [True, False])
def test_generate_maximal_network(
    toluene_vs_others,
    with_progress,
    with_scorer,
    extra_mapper,
    lomap_old_mapper,
    simple_scorer,
):
    toluene, others = toluene_vs_others

    if extra_mapper:
        mappers = [lomap_old_mapper, BadMapper()]
    else:
        mappers = lomap_old_mapper

    scorer = simple_scorer if with_scorer else None

    network = openfe.setup.ligand_network_planning.generate_maximal_network(
        ligands=others + [toluene],
        mappers=mappers,
        scorer=scorer,
        progress=with_progress,
    )

    expected_names = set([c.name for c in others] + ["toluene"])

    assert len(network.nodes) == len(expected_names)

    # check that all ligands are present, i.e. we included everyone
    ligands_in_network = {mol.name for mol in network.nodes}
    assert ligands_in_network == expected_names

    if extra_mapper:
        # two edges per pair of nodes, one for each mapper
        edge_count = len(expected_names) * (len(expected_names) - 1)
    else:
        # one edge per pair of nodes
        edge_count = (len(expected_names) * (len(expected_names) - 1)) / 2

    assert len(network.edges) == edge_count

    if with_scorer:
        for edge in network.edges:
            score = edge.annotations["score"]
            assert score == 1 - 1 / len(edge.componentA_to_componentB)
    else:
        for edge in network.edges:
            assert "score" not in edge.annotations


class TestMinimalSpanningNetworkGenerator:
    @pytest.mark.parametrize("multi_mappers", [False, True])
    def test_minimal_spanning_network(self, toluene_vs_others, multi_mappers, lomap_old_mapper, simple_scorer):
        toluene, others = toluene_vs_others
        ligands = [toluene] + others

        if multi_mappers:
            mappers = [BadMapper(), lomap_old_mapper]
        else:
            mappers = lomap_old_mapper

        scorer = simple_scorer

        network = openfe.ligand_network_planning.generate_minimal_spanning_network(
            ligands=ligands,
            mappers=mappers,
            scorer=scorer,
        )

        expected_names = {c.name for c in ligands}

        # couple sanity checks
        assert len(network.nodes) == len(expected_names)
        assert len(network.edges) == len(expected_names) - 1
        assert network.is_connected()

        # check that all ligands are present, i.e. we included everyone
        ligands_in_network = {mol.name for mol in network.nodes}
        assert ligands_in_network == expected_names

        for edge in network.edges:
            # make sure we didn't take the bad mapper, which would always be a length of 1 ({0:0})
            assert len(edge.componentA_to_componentB) > 1
            assert 'score' in edge.annotations
            assert edge.annotations['score'] == 1 - 1 / len(edge.componentA_to_componentB)

    def test_minimal_spanning_network_no_scorer_error(self, toluene_vs_others, lomap_old_mapper):
        """Expect a KeyError if no scorer is passed."""
        # NOTE: I'm not making this error handling prettier until the konnektor integration
        toluene, others = toluene_vs_others
        ligands = [toluene] + others

        with pytest.raises(KeyError, match="score"):
            _ = openfe.ligand_network_planning.generate_minimal_spanning_network(
                ligands=ligands,
                mappers=lomap_old_mapper,
                scorer=None,
            )

    def test_minimal_spanning_network_connectedness(self, deterministic_minimal_spanning_network):
        # makes sure we don't have duplicate edges?
        found_pairs = set()
        for edge in deterministic_minimal_spanning_network.edges:
            pair = frozenset([edge.componentA, edge.componentB])
            assert pair not in found_pairs
            found_pairs.add(pair)

        assert deterministic_minimal_spanning_network.is_connected()

    def test_minimal_spanning_network_regression(self, deterministic_minimal_spanning_network):
        """issue #244, this was previously giving non-reproducible (yet valid) networks when scores were tied."""
        edge_names = {(e.componentA.name, e.componentB.name) for e in deterministic_minimal_spanning_network.edges}
        expected_edge_names = {
            ('1,3,7-trimethylnaphthalene', '2,6-dimethylnaphthalene'),
            ('1-butyl-4-methylbenzene', '2-methyl-6-propylnaphthalene'),
            ('2,6-dimethylnaphthalene', '2-methyl-6-propylnaphthalene'),
            ('2,6-dimethylnaphthalene', '2-methylnaphthalene'),
            ('2,6-dimethylnaphthalene', '2-naftanol'),
            ('2,6-dimethylnaphthalene', 'methylcyclohexane'),
            ('2,6-dimethylnaphthalene', 'toluene'),
        }
        assert len(deterministic_minimal_spanning_network.nodes) == 8
        assert len(edge_names) == len(expected_edge_names)
        assert edge_names == expected_edge_names

    def test_minimal_spanning_network_unreachable(self, toluene_vs_others, lomap_old_mapper, simple_scorer):
        toluene, others = toluene_vs_others
        nimrod = openfe.SmallMoleculeComponent(mol_from_smiles("N"), name='nimrod')

        scorer = simple_scorer

        with pytest.raises(RuntimeError, match=r"Unable to create edges to some nodes: \[SmallMoleculeComponent\(name=nimrod\)\]"):
            _ = openfe.setup.ligand_network_planning.generate_minimal_spanning_network(
                ligands=others + [toluene, nimrod],
                mappers=[lomap_old_mapper],
                scorer=scorer,
            )


class TestMinimalRedundantNetworkGenerator:
    def test_minimal_redundant_network(self, deterministic_minimal_redundant_network, toluene_vs_others):
        toluene, others = toluene_vs_others
        ligands = [toluene] + others
        expected_names = {c.name for c in ligands}

        # check that all ligands are present, i.e. we included everyone
        assert len(deterministic_minimal_redundant_network.nodes) == len(ligands)
        ligands_in_network = {mol.name for mol in deterministic_minimal_redundant_network.nodes}
        assert ligands_in_network == expected_names

        # we expect double the number of edges of an mst
        assert len(deterministic_minimal_redundant_network.edges) == 2 * (len(ligands) - 1)

        for edge in deterministic_minimal_redundant_network.edges:
            # lomap should find something
            assert edge.componentA_to_componentB != {0: 0}

    def test_minimal_redundant_network_connectedness(self, deterministic_minimal_redundant_network):
        # makes sure we don't have duplicate edges?

        found_pairs = set()
        for edge in deterministic_minimal_redundant_network.edges:
            pair = frozenset([edge.componentA, edge.componentB])
            assert pair not in found_pairs
            found_pairs.add(pair)

        assert deterministic_minimal_redundant_network.is_connected()

    def test_redundant_vs_spanning_network(
        self,
        toluene_vs_others,
        lomap_old_mapper,
        deterministic_toluene_mst_scorer,
        deterministic_minimal_spanning_network,
    ):
        """when setting minimal redundant network to only take one MST, it should be equivalent to the base MST."""

        toluene, others = toluene_vs_others
        scorer = deterministic_toluene_mst_scorer

        minimal_redundant_network = openfe.setup.ligand_network_planning.generate_minimal_redundant_network(
            ligands=others + [toluene],
            mappers=lomap_old_mapper,
            scorer=scorer,
            mst_num=1
        )
        assert deterministic_minimal_spanning_network.edges == minimal_redundant_network.edges

    def test_minimal_redundant_network_edges(self, deterministic_minimal_redundant_network):
        """issue #244, this was previously giving non-reproducible (yet valid)
        networks when scores were tied."""
        edge_names = {(e.componentA.name, e.componentB.name) for e in deterministic_minimal_redundant_network.edges}
        expected_names = {
            ('1,3,7-trimethylnaphthalene', '2,6-dimethylnaphthalene'),
            ('1,3,7-trimethylnaphthalene', '2-methyl-6-propylnaphthalene'),
            ('1-butyl-4-methylbenzene', '2,6-dimethylnaphthalene'),
            ('1-butyl-4-methylbenzene', '2-methyl-6-propylnaphthalene'),
            ('1-butyl-4-methylbenzene', 'toluene'),
            ('2,6-dimethylnaphthalene', '2-methyl-6-propylnaphthalene'),
            ('2,6-dimethylnaphthalene', '2-methylnaphthalene'),
            ('2,6-dimethylnaphthalene', '2-naftanol'),
            ('2,6-dimethylnaphthalene', 'methylcyclohexane'),
            ('2,6-dimethylnaphthalene', 'toluene'),
            ('2-methyl-6-propylnaphthalene', '2-methylnaphthalene'),
            ('2-methylnaphthalene', '2-naftanol'),
            ('2-methylnaphthalene', 'methylcyclohexane'),
            ('2-methylnaphthalene', 'toluene')
        }

        assert len(edge_names) == len(expected_names)
        assert edge_names == expected_names

    def test_minimal_redundant_network_redundant(self, deterministic_minimal_redundant_network):
        """test that each node is connected to 2 edges"""
        network = deterministic_minimal_redundant_network
        for node in network.nodes:
            assert (
                len(network.graph.in_edges(node)) + len(network.graph.out_edges(node))
                >= 2
            )

    def test_minimal_redundant_network_unreachable(self, toluene_vs_others, lomap_old_mapper, simple_scorer):
        toluene, others = toluene_vs_others
        nimrod = openfe.SmallMoleculeComponent(mol_from_smiles("N"), name='nimrod')

        scorer = simple_scorer

        with pytest.raises(RuntimeError, match=r"Unable to create edges to some nodes: \[SmallMoleculeComponent\(name=nimrod\)\]"):
            _ = openfe.setup.ligand_network_planning.generate_minimal_redundant_network(
                ligands=others + [toluene, nimrod],
                mappers=[lomap_old_mapper],
                scorer=scorer
            )

class TestGenerateNetworkFromNames:
    def test_generate_network_from_names(self, atom_mapping_basic_test_files, lomap_old_mapper):
        ligands = list(atom_mapping_basic_test_files.values())

        requested = [
            ('toluene', '2-naftanol'),
            ('2-methylnaphthalene', '2-naftanol'),
        ]

        network = openfe.setup.ligand_network_planning.generate_network_from_names(
            ligands=ligands,
            names=requested,
            mapper=lomap_old_mapper,
        )

        expected_node_names = {c.name for c in ligands}
        actual_node_names = {n.name for n in network.nodes}

        assert len(network.nodes) == len(ligands)
        assert actual_node_names == expected_node_names

        assert len(network.edges) == 2
        actual_edges = {(e.componentA.name, e.componentB.name) for e in network.edges}
        assert set(requested) == actual_edges

    def test_generate_network_from_names_bad_name_error(self, atom_mapping_basic_test_files, lomap_old_mapper):
        ligands = list(atom_mapping_basic_test_files.values())

        requested = [
            ('hank', '2-naftanol'),
            ('2-methylnaphthalene', '2-naftanol'),
        ]

        with pytest.raises(KeyError, match=r"Invalid name\(s\) requested \['hank'\]."):
            _ = openfe.setup.ligand_network_planning.generate_network_from_names(
                ligands=ligands,
                names=requested,
                mapper=lomap_old_mapper,
            )


    def test_generate_network_from_names_duplicate_name(self, atom_mapping_basic_test_files, lomap_old_mapper):
        ligands = list(atom_mapping_basic_test_files.values())
        ligands = ligands + [ligands[0]]

        requested = [
            ('toluene', '2-naftanol'),
            ('2-methylnaphthalene', '2-naftanol'),
        ]

        with pytest.raises(ValueError, match=r"Duplicate names: \['1,3,7-trimethylnaphthalene'\]"):
            _ = openfe.setup.ligand_network_planning.generate_network_from_names(
                ligands=ligands,
                names=requested,
                mapper=lomap_old_mapper,
            )

class TestNetworkFromIndices:
    def test_network_from_indices(self, atom_mapping_basic_test_files, lomap_old_mapper):
        ligands = list(atom_mapping_basic_test_files.values())

        requested = [(0, 1), (2, 3)]

        network = openfe.setup.ligand_network_planning.generate_network_from_indices(
            ligands=ligands,
            indices=requested,
            mapper=lomap_old_mapper,
        )

        assert len(network.nodes) == len(ligands)
        assert len(network.edges) == 2

        edges = list(network.edges)
        expected_edges = {(ligands[0], ligands[1]), (ligands[2], ligands[3])}
        actual_edges = {
            (edges[0].componentA, edges[0].componentB),
            (edges[1].componentA, edges[1].componentB),
        }

        assert actual_edges == expected_edges

    def test_network_from_indices_indexerror(self, atom_mapping_basic_test_files, lomap_old_mapper):
        ligands = list(atom_mapping_basic_test_files.values())

        requested = [(20, 1), (2, 3)]

        with pytest.raises(IndexError, match="Invalid ligand id"):
            _ = openfe.setup.ligand_network_planning.generate_network_from_indices(
                ligands=ligands,
                indices=requested,
                mapper=lomap_old_mapper,
            )

    def test_network_from_indices_disconnected_warning(
        self, atom_mapping_basic_test_files, lomap_old_mapper
    ):
        ligands = list(atom_mapping_basic_test_files.values())
        requested = [(0, 1), (1, 2)]

        with pytest.warns(UserWarning):
            _ = openfe.setup.ligand_network_planning.generate_network_from_indices(
                ligands=ligands,
                indices=requested,
                mapper=lomap_old_mapper,
            )


@pytest.mark.parametrize(
    "file_fixture, loader",
    [
        ["orion_network", openfe.setup.ligand_network_planning.load_orion_network],
        ["fepplus_network", openfe.setup.ligand_network_planning.load_fepplus_network],
    ],
)
def test_network_from_external(file_fixture, loader, request, benzene_modifications):

    network_file = request.getfixturevalue(file_fixture)

    network = loader(
        ligands=[l for l in benzene_modifications.values()],
        mapper=openfe.LomapAtomMapper(),
        network_file=network_file,
    )

    expected_edges = {
        (benzene_modifications["benzene"], benzene_modifications["toluene"]),
        (benzene_modifications["benzene"], benzene_modifications["phenol"]),
        (benzene_modifications["benzene"], benzene_modifications["benzonitrile"]),
        (benzene_modifications["benzene"], benzene_modifications["anisole"]),
        (benzene_modifications["benzene"], benzene_modifications["styrene"]),
        (benzene_modifications["benzene"], benzene_modifications["benzaldehyde"]),
    }

    actual_edges = {(e.componentA, e.componentB) for e in list(network.edges)}

    assert len(network.nodes) == 7
    assert len(network.edges) == 6
    assert actual_edges == expected_edges


@pytest.mark.parametrize(
    "file_fixture, loader",
    [
        ["orion_network", openfe.setup.ligand_network_planning.load_orion_network],
        ["fepplus_network", openfe.setup.ligand_network_planning.load_fepplus_network],
    ],
)
def test_network_from_external_unknown_edge(file_fixture, loader, request,
                                            benzene_modifications):
    network_file = request.getfixturevalue(file_fixture)
    ligands = [l for l in benzene_modifications.values() if l.name != 'phenol']

    with pytest.raises(KeyError, match="Invalid name"):
        _ = loader(
            ligands=ligands,
            mapper=openfe.LomapAtomMapper(),
            network_file=network_file,
        )


BAD_ORION_NETWORK = """\
# Total number of edges: 6
# ------------------------
benzene >>> toluene
benzene >> phenol
benzene >> benzonitrile
benzene >> anisole
benzene >> styrene
benzene >> benzaldehyde
"""


def test_bad_orion_network(benzene_modifications, tmpdir):
    with tmpdir.as_cwd():
        with open('bad_orion_net.dat', 'w') as f:
            f.write(BAD_ORION_NETWORK)

        with pytest.raises(KeyError, match="line does not match"):
            _ = openfe.setup.ligand_network_planning.load_orion_network(
                ligands=[l for l in benzene_modifications.values()],
                mapper=openfe.LomapAtomMapper(),
                network_file='bad_orion_net.dat',
            )


BAD_EDGES = """\
1c91235:9c91235 benzene -> toluene
1c91235:7876633 benzene -> phenol
1c91235:2a51f95 benzene -> benzonitrile
1c91235:efja0bc benzene -> anisole
1c91235:7877722 benzene -> styrene
1c91235:99930cd benzene -> benzaldehyde
"""


def test_bad_edges_network(benzene_modifications, tmpdir):
    with tmpdir.as_cwd():
        with open('bad_edges.edges', 'w') as f:
            f.write(BAD_EDGES)

        with pytest.raises(KeyError, match="line does not match"):
            _ = openfe.setup.ligand_network_planning.load_fepplus_network(
                ligands=[l for l in benzene_modifications.values()],
                mapper=openfe.LomapAtomMapper(),
                network_file='bad_edges.edges',
            )
