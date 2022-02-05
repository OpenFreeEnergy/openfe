from dataclasses import dataclass
from typing import Iterable
import pytest

from openfe.setup import Molecule, AtomMapping, Network
from rdkit import Chem

@dataclass
class _NetworkTestContainer:
    """Container to facilitate network testing"""
    network : Network
    nodes : Iterable[Molecule]
    edges : Iterable[AtomMapping]

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)


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
    )


_NETWORK_NAMES = ['simple', 'doubled_edge', 'singleton_node',]

@pytest.fixture
def all_networks(
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

    # assertion fails if there is a mismatch between _NETWORK_NAMES and the
    # keys in the dict by this fixture (error in test suite implementation)
    assert set(network_dct) == set(_NETWORK_NAMES)

    return network_dct


class TestNetwork:
    @pytest.mark.parametrize('network_name', _NETWORK_NAMES)
    def test_graph(self, all_networks, network_name):
        # The NetworkX graph that comes from the ``.graph`` property should
        # have nodes and edges that match the Network container object.
        container = all_networks[network_name]
        network = container.network
        pytest.skip("TODO")

    @pytest.mark.parametrize('network_name', _NETWORK_NAMES)
    def test_graph_immutability(self, all_networks, network_name):
        # The NetworkX graph that comes from that ``.graph`` property should
        # be immutable.
        container = all_networks[network_name]
        network = container.network
        pytest.skip("TODO")

    @pytest.mark.parametrize('network_name', _NETWORK_NAMES)
    def test_nodes(self, all_networks, network_name):
        # The nodes reported by a ``Network`` should match expectations for
        # that network.
        container = all_networks[network_name]
        network = container.network
        assert len(network.nodes) == container.n_nodes
        assert set(network.nodes) == set(container.nodes)

    @pytest.mark.parametrize('network_name', _NETWORK_NAMES)
    def test_edges(self, all_networks, network_name):
        # The edges reported by a ``Network`` should match expectations for
        # that network.
        container = all_networks[network_name]
        network = container.network
        assert len(network.edges) == container.n_edges
        assert set(network.edges) == set(container.edges)

    def test_enlarge_graph(self, simple_network):
        pytest.skip("TODO")
