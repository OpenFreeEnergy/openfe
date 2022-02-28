from typing import Iterable, NamedTuple
import pytest

from openfe.setup import Molecule, AtomMapping, Network
from rdkit import Chem

from networkx import NetworkXError


class _NetworkTestContainer(NamedTuple):
    """Container to facilitate network testing"""
    network: Network
    nodes: Iterable[Molecule]
    edges: Iterable[AtomMapping]
    n_nodes: int
    n_edges: int


@pytest.fixture
def mols():
    mol1 = Molecule(Chem.MolFromSmiles("CCO"))
    mol2 = Molecule(Chem.MolFromSmiles("CC"))
    mol3 = Molecule(Chem.MolFromSmiles("CO"))
    return mol1, mol2, mol3


@pytest.fixture
def std_edges(mols):
    mol1, mol2, mol3 = mols
    edge12 = AtomMapping(mol1, mol2, {0: 0, 1: 1})
    edge23 = AtomMapping(mol2, mol3, {0: 0})
    edge13 = AtomMapping(mol1, mol3, {0: 0, 2: 1})
    return edge12, edge23, edge13


@pytest.fixture
def simple_network(mols, std_edges):
    """Network with no edges duplicated and all nodes in edges"""
    network = Network(std_edges)
    return _NetworkTestContainer(
        network=network,
        nodes=mols,
        edges=std_edges,
        n_nodes=3,
        n_edges=3,
    )


@pytest.fixture
def doubled_edge_network(mols, std_edges):
    """Network with more than one edge associated with a node pair"""
    mol1, mol2, mol3 = mols
    extra_edge = AtomMapping(mol1, mol3, {0: 0, 1: 1})
    edges = list(std_edges) + [extra_edge]
    return _NetworkTestContainer(
        network=Network(edges),
        nodes=mols,
        edges=edges,
        n_nodes=3,
        n_edges=4,
    )


@pytest.fixture
def singleton_node_network(mols, std_edges):
    """Network with nodes not included in any edges"""
    extra_mol = Molecule(Chem.MolFromSmiles("CCC"))
    all_mols = list(mols) + [extra_mol]
    return _NetworkTestContainer(
        network=Network(edges=std_edges, nodes=all_mols),
        nodes=all_mols,
        edges=std_edges,
        n_nodes=4,
        n_edges=3,
    )


@pytest.fixture(params=['simple', 'doubled_edge', 'singleton_node'])
def network_container(
    request,
    simple_network,
    doubled_edge_network,
    singleton_node_network,
):
    """Fixture to allow parameterization of the network test"""
    network_dct = {
        'simple': simple_network,
        'doubled_edge': doubled_edge_network,
        'singleton_node': singleton_node_network,
    }
    return network_dct[request.param]


class TestNetwork:
    def test_node_type(self, network_container):
        n = network_container.network

        assert all((isinstance(node, Molecule) for node in n.nodes))

    def test_graph(self, network_container):
        # The NetworkX graph that comes from the ``.graph`` property should
        # have nodes and edges that match the Network container object.
        graph = network_container.network.graph
        assert len(graph.nodes) == network_container.n_nodes
        assert set(graph.nodes) == set(network_container.nodes)
        assert len(graph.edges) == network_container.n_edges
        # extract the AtomMappings from the nx edges
        mappings = [
            atommapping for _, _, atommapping in graph.edges.data('object')
        ]
        assert set(mappings) == set(network_container.edges)
        # ensure AtomMapping stored in nx edge is consistent with nx edge
        for mol1, mol2, atommapping in graph.edges.data('object'):
            assert atommapping.mol1 == mol1
            assert atommapping.mol2 == mol2

    def test_graph_immutability(self, mols, network_container):
        # The NetworkX graph that comes from that ``.graph`` property should
        # be immutable.
        graph = network_container.network.graph
        mol = Molecule(Chem.MolFromSmiles("CCCC"))
        mol_CC = mols[1]
        edge = AtomMapping(mol, mol_CC, {1: 0, 2: 1})
        with pytest.raises(NetworkXError, match="Frozen graph"):
            graph.add_node(mol)
        with pytest.raises(NetworkXError, match="Frozen graph"):
            graph.add_edge(edge)

    def test_nodes(self, network_container):
        # The nodes reported by a ``Network`` should match expectations for
        # that network.
        network = network_container.network
        assert len(network.nodes) == network_container.n_nodes
        assert set(network.nodes) == set(network_container.nodes)

    def test_edges(self, network_container):
        # The edges reported by a ``Network`` should match expectations for
        # that network.
        network = network_container.network
        assert len(network.edges) == network_container.n_edges
        assert set(network.edges) == set(network_container.edges)

    def test_enlarge_graph_add_nodes(self, simple_network):
        # New nodes added via ``enlarge_graph`` should exist in the newly
        # created network
        new_mol = Molecule(Chem.MolFromSmiles("CCCC"))
        network = simple_network.network
        new_network = network.enlarge_graph(nodes=[new_mol])
        assert new_network is not network
        assert new_mol not in network.nodes
        assert new_mol in new_network.nodes
        assert len(new_network.nodes) == len(network.nodes) + 1
        assert set(new_network.edges) == set(network.edges)

    def test_enlarge_graph_add_edges(self, mols, simple_network):
        # New edges added via ``enlarge_graph`` should exist in the newly
        # created network
        mol1, _, mol3 = mols
        extra_edge = AtomMapping(mol1, mol3, {0: 0, 1: 1})
        network = simple_network.network
        new_network = network.enlarge_graph(edges=[extra_edge])
        assert new_network is not network
        assert extra_edge not in network.edges
        assert extra_edge in new_network.edges
        assert len(new_network.edges) == len(network.edges) + 1
        assert set(new_network.nodes) == set(network.nodes)

    def test_enlarge_graph_add_edges_new_nodes(self, mols, simple_network):
        # New nodes included implicitly by edges added in ``enlarge_graph``
        # should exist in the newly created network
        new_mol = Molecule(Chem.MolFromSmiles("CCCC"))
        mol_CC = mols[1]
        extra_edge = AtomMapping(new_mol, mol_CC, {1: 0, 2: 1})
        network = simple_network.network
        new_network = network.enlarge_graph(edges=[extra_edge])
        assert new_network is not network
        assert extra_edge not in network.edges
        assert extra_edge in new_network.edges
        assert new_mol not in network.nodes
        assert new_mol in new_network.nodes
        assert len(new_network.edges) == len(network.edges) + 1
        assert len(new_network.nodes) == len(network.nodes) + 1

    def test_enlarge_graph_add_duplicate_node(self, simple_network):
        # Adding a duplicate of an existing node should create a new network
        # with the same edges and nodes as the previous one.
        duplicate = Molecule(Chem.MolFromSmiles("CC"))
        network = simple_network.network

        existing = network.nodes
        assert duplicate in existing  # matches by ==
        for mol in existing:
            # the duplicate is, in fact, not the same object in memory
            assert mol is not duplicate

        new_network = network.enlarge_graph(nodes=[duplicate])
        assert len(new_network.nodes) == len(network.nodes)
        assert set(new_network.nodes) == set(network.nodes)
        assert len(new_network.edges) == len(network.edges)
        assert set(new_network.edges) == set(network.edges)

        for mol in new_network.nodes:
            # not sure is this needs to be formally required
            assert mol is not duplicate

    def test_enlarge_graph_add_duplicate_edge(self, mols, simple_network):
        # Adding a duplicate of an existing edge should create a new network
        # with the same edges and nodes as the previous one.
        mol1, _, mol3 = mols
        duplicate = AtomMapping(mol1, mol3, {0: 0, 2: 1})
        network = simple_network.network

        existing = network.edges
        assert duplicate in existing  # matches by ==
        for edge in existing:
            # duplicate is not the same object in memory
            assert duplicate is not edge

        new_network = network.enlarge_graph(edges=[duplicate])
        assert len(new_network.nodes) == len(network.nodes)
        assert set(new_network.nodes) == set(network.nodes)
        assert len(new_network.edges) == len(network.edges)
        assert set(new_network.edges) == set(network.edges)

        for edge in new_network.edges:
            # not sure is this needs to be formally required
            assert edge is not duplicate

    def test_serialization_cycle(self, simple_network):
        network = simple_network.network
        serialized = network.to_graphml()
        deserialized = Network.from_graphml(serialized)
        reserialized = deserialized.to_graphml()
        assert serialized == reserialized
        assert network == deserialized

    def test_to_graphml(self, simple_network, serialization_template):
        expected = serialization_template("network_template.graphml")
        assert simple_network.network.to_graphml() + "\n" == expected

    def test_from_graphml(self, simple_network, serialization_template):
        contents = serialization_template("network_template.graphml")
        assert Network.from_graphml(contents) == simple_network.network
