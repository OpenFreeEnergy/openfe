# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

from typing import Iterable, Callable
import itertools

import networkx as nx

from gufe import SmallMoleculeComponent
from .abstract_network_planner import AbstractRelativeLigandNetworkPlanner

from openfe.setup import LigandNetwork
from openfe.setup.atom_mapping import LigandAtomMapper, LigandAtomMapping


class MinimalSpanningNetworkPlanner(AbstractRelativeLigandNetworkPlanner):
    def __init__(self, mappers: Iterable[LigandAtomMapper], mapping_scorer=None):
        """

        Parameters
        ----------
        mappers : iterable of LigandAtomMappers
          mappers to use, at least 1 required
        mapping_scorer : scoring function, optional
          a callable which returns a float for any LigandAtomMapping.  Used to
          assign scores to potential mappings, higher scores indicate worse
          mappings.
        """
        self._mappers = mappers
        self._mapping_scorer = mapping_scorer

    def __call__(self, ligands: Iterable[SmallMoleculeComponent]) -> LigandNetwork:
        """Plan a Network which connects all ligands with minimal total score

        Parameters
        ----------
        ligands : Iterable[SmallMoleculeComponent]
          the ligands to include in the Network

        """

        nodes = list(ligands)

        # First create a network with all the proposed mappings (scored)
        mapping_generator = itertools.chain.from_iterable(
            mapper.suggest_mappings(molA, molB)
            for molA, molB in itertools.combinations(nodes, 2)
            for mapper in self._mappers
        )
        self._mappings = [
            mapping.with_annotations({"score": self._mapping_scorer(mapping)})
            for mapping in mapping_generator
        ]
        network = LigandNetwork(self.mappings, nodes=nodes)

        # Next analyze that network to create minimal spanning network. Because
        # we carry the original (directed) LigandAtomMapping, we don't lose
        # direction information when converting to an undirected graph.
        min_edges = nx.minimum_spanning_edges(
            nx.MultiGraph(network.graph), weight="score"
        )
        min_mappings = [edge_data["object"] for _, _, _, edge_data in min_edges]
        min_network = LigandNetwork(min_mappings)
        missing_nodes = set(nodes) - set(min_network.nodes)
        if missing_nodes:
            raise RuntimeError(
                "Unable to create edges to some nodes: " + str(list(missing_nodes))
            )

        return min_network
