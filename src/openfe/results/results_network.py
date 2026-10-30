# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from collections.abc import Iterable, Self
from functools import cache

import networkx as nx
from gufe import AlchemicalArchive, ChemicalSystem, TransformationBase
from gufe.protocols import Protocol, ProtocolDAGResult
from gufe.tokenizables import GufeTokenizable


class TransformationResult(GufeTokenizable):
    _protocol_dag_results: list[ProtocolDAGResult]
    _protocol: Protocol
    _transformation: TransformationBase

    def __init__(self, protocol_dag_results):
        pass

    @property
    def stateA(self) -> ChemicalSystem:
        return self._stateA

    @property
    def stateB(self) -> ChemicalSystem:
        return self._stateB

    @property
    def protocol_dag_results(self) -> list[ProtocolDAGResult]:
        return self._protocol_dag_results

    @property
    def transformation(self) -> TransformationBase:
        return self._transformation


class ResultsNetwork:
    # NOTE: this is a class with mutable edges and nodes for the prototyping phase. we may decide to make this a gufe tokenizable or minimally make the nodes & edges tuples
    _name: str | None
    _edges: list[TransformationResult]
    _graph: nx.MultiDiGraph | None

    """A class for storing and inspecting results from openfe execution."""

    def __init__(self, results: Iterable[tuple[ProtocolDAGResult]], name: str | None = None):
        self._name = name
        self._edges: list[TransformationResult] = []

    @staticmethod
    def _generate_graph(edges) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph()
        for e in edges:
            g.add_edge(e.stateA, e.stateB)
        # TODO: do we want the option to add lone nodes so that a disconnected graph is apparent?
        return nx.freeze(g)

    @cache
    @property
    def graph(self) -> nx.MultiDiGraph:
        """networkx graph representation of the ResultsNetwork"""
        graph = self._generate_graph(self._edges, self._nodes)
        return graph.copy(as_view=True)

    @property
    def nodes(self) -> Iterable[ChemicalSystem]: ...

    @property
    def edges(self) -> Iterable[TransformationResult]:
        pass

    @property
    def name(self) -> str | None:
        """Optional identifier for the network."""
        return self._name

    def get_results_by_protocol(self) -> dict[Protocol, ProtocolDAGResult]:
        """Returns a dict of protocols and the edges that use them."""
        raise NotImplementedError

    @classmethod
    def from_warehouse(cls, warehouse) -> Self:
        """Load a ResultsNetwork from an openfe Warehouse."""
        raise NotImplementedError

    @classmethod
    def to_alchemical_archive(self) -> AlchemicalArchive:
        """Store the ResultsNetwork as an AlchemicalArchive
        (Note: a valid ResultsNetwork may not be a valid AlchemicalArchive)
        """
        raise NotImplementedError
