from typing import List, TypeVar, Iterable
from openfe.setup import AtomMapping, Molecule

import networkx as nx
NetworkType = TypeVar('Network')


class Network:
    """Network container

    Parameters
    ----------
    edges : Iterable[AtomMapping]
        edges for this network
    nodes : Iterable[Molecule]
        nodes for this network
    """
    def __init__(
        self, edges: Iterable[AtomMapping], nodes: Iterable[Molecule] = None
    ):
        if nodes is None:
            nodes = []

        self._edges = list(set(edges))
        edge_nodes  = set.union(*[{edge.mol1, edge.mol2} for edge in edges])
        self._nodes = list(edge_nodes | set(nodes))
        self._graph = None

    @property
    def graph(self) -> nx.Graph:
        """NetworkX graph for this network"""
        if self._graph is None:
            graph = nx.MultiDiGraph()
            for edge in self._edges:
                graph.add_edge(edge.mol1, edge.mol2, object=edge)

            self._graph = nx.freeze(graph)

        return self._graph

    @property
    def edges(self) -> List[AtomMapping]:
        """List of edges"""
        return self._edges

    @property
    def nodes(self) -> List[Molecule]:
        """List of nodes"""
        return self._nodes

    def enlarge_graph(self, *, edges=None, nodes=None) -> NetworkType:
        """
        Create a new network with the edge added

        Parameters
        ----------
        edge : :class:`.AtomMapping`
            the edge to append to this network

        Returns
        -------
        :class:`.Network :
            a new network adding the given edge to this network
        """
        if edges is None:
            edges = []

        if nodes is None:
            nodes = []

        return Network(self.edges + edges, self.nodes + nodes)

    def annotate_node(self, node, annotation) -> NetworkType:
        """Return a new network with the additional node annotation"""
        raise NotImplementedError("Waiting on annotations")

    def annotate_edge(self, edge, annotation) -> NetworkType:
        """Return a new network with the additional edge annotation"""
        raise NotImplementedError("Waiting on annotations")
