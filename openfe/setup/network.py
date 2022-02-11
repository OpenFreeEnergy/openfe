from __future__ import annotations

from typing import FrozenSet, TypeVar, Iterable
from openfe.setup import AtomMapping, Molecule

import networkx as nx


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

        self._edges = frozenset(edges)
        edge_nodes = set.union(*[{edge.mol1, edge.mol2} for edge in edges])
        self._nodes = frozenset(edge_nodes) | frozenset(nodes)
        self._graph = None

    @property
    def graph(self) -> nx.Graph:
        """NetworkX graph for this network"""
        if self._graph is None:
            graph = nx.MultiDiGraph()
            for node in self._nodes:
                graph.add_node(node)
            for edge in self._edges:
                graph.add_edge(edge.mol1, edge.mol2, object=edge)

            self._graph = nx.freeze(graph)

        return self._graph

    @property
    def edges(self) -> FrozenSet[AtomMapping]:
        """A read-only view of the edges of the Network"""
        return self._edges

    @property
    def nodes(self) -> FrozenSet[Molecule]:
        """A read-only view of the nodes of the Network"""
        return self._nodes

    def enlarge_graph(self, *, edges=None, nodes=None) -> Network:
        """
        Create a new network with the given edges and nodes added

        Parameters
        ----------
        edges : Iterable[:class:`.AtomMapping`]
            edges to append to this network
        nodes : Iterable[:class:`.Molecule`]
            nodes to append to this network

        Returns
        -------
        :class:`.Network :
            a new network adding the given edges and nodes to this network
        """
        if edges is None:
            edges = set([])

        if nodes is None:
            nodes = set([])

        return Network(self.edges | set(edges), self.nodes | set(nodes))

    def annotate_node(self, node, annotation) -> Network:
        """Return a new network with the additional node annotation"""
        raise NotImplementedError("Waiting on annotations")

    def annotate_edge(self, edge, annotation) -> Network:
        """Return a new network with the additional edge annotation"""
        raise NotImplementedError("Waiting on annotations")
