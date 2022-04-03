# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
from __future__ import annotations
import json

from typing import FrozenSet, Iterable

from openfe.setup import LigandAtomMapping, SmallMoleculeComponent
import openfe

import networkx as nx


class Network:
    """A directed graph connecting many ligands according to their atom mapping

    Parameters
    ----------
    edges : Iterable[LigandAtomMapping]
        edges for this network
    nodes : Iterable[SmallMoleculeComponent]
        nodes for this network
    """
    def __init__(
        self,
        edges: Iterable[LigandAtomMapping],
        nodes: Iterable[SmallMoleculeComponent] = None
    ):
        if nodes is None:
            nodes = []

        self._edges = frozenset(edges)
        edge_nodes = set.union(*[{edge.molA, edge.molB} for edge in edges])
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
                graph.add_edge(edge.molA, edge.molB, object=edge)

            self._graph = nx.freeze(graph)

        return self._graph

    @property
    def edges(self) -> FrozenSet[LigandAtomMapping]:
        """A read-only view of the edges of the Network"""
        return self._edges

    @property
    def nodes(self) -> FrozenSet[SmallMoleculeComponent]:
        """A read-only view of the nodes of the Network"""
        return self._nodes

    def __eq__(self, other):
        return self.nodes == other.nodes and self.edges == other.edges

    def _serializable_graph(self) -> nx.Graph:
        """
        Create NetworkX graph with serializable attribute representations.

        This enables us to use easily use different serialization
        approaches.
        """
        # sorting ensures that we always preserve order in files, so two
        # identical networks will show no changes if you diff their
        # serialized versions
        sorted_nodes = sorted(self.nodes, key=lambda m: (m.smiles, m.name))
        mol_to_label = {mol: f"mol{num}"
                        for num, mol in enumerate(sorted_nodes)}

        edge_data = sorted([
            (
                mol_to_label[edge.molA],
                mol_to_label[edge.molB],
                json.dumps(list(edge.molA_to_molB.items()))
            )
            for edge in self.edges
        ])

        # from here, we just build the graph
        serializable_graph = nx.MultiDiGraph(ofe_version=openfe.__version__)
        for mol, label in mol_to_label.items():
            serializable_graph.add_node(label, sdf=mol.to_sdf())

        for molA, molB, mapping in edge_data:
            serializable_graph.add_edge(molA, molB, mapping=mapping)

        return serializable_graph

    @classmethod
    def _from_serializable_graph(cls, graph: nx.Graph):
        """Create network from NetworkX graph with serializable attributes.

        This is the inverse of ``_serializable_graph``.
        """
        label_to_mol = {node: SmallMoleculeComponent.from_sdf_string(sdf)
                        for node, sdf in graph.nodes(data='sdf')}

        edges = [
            LigandAtomMapping(molA=label_to_mol[node1],
                              molB=label_to_mol[node2],
                              molA_to_molB=dict(json.loads(mapping)))
            for node1, node2, mapping in graph.edges(data='mapping')
        ]

        return cls(edges=edges, nodes=label_to_mol.values())

    def to_graphml(self) -> str:
        """Return the GraphML string representing this ``Network``.

        This is the primary serialization mechanism for this class.

        Returns
        -------
        str :
            string representing this network in GraphML format
        """
        return "\n".join(nx.generate_graphml(self._serializable_graph()))

    @classmethod
    def from_graphml(cls, graphml_str: str):
        """Create ``Network`` from GraphML string.

        This is the primary deserialization mechanism for this class.

        Parameters
        ----------
        graphml_str : str
            GraphML string representation of a :class:`.Network`

        Returns
        -------
        :class:`.Network`:
            new network from the GraphML
        """
        return cls._from_serializable_graph(nx.parse_graphml(graphml_str))

    def enlarge_graph(self, *, edges=None, nodes=None) -> Network:
        """
        Create a new network with the given edges and nodes added

        Parameters
        ----------
        edges : Iterable[:class:`.LigandAtomMapping`]
            edges to append to this network
        nodes : Iterable[:class:`.SmallMoleculeComponent`]
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
