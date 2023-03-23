from typing import Iterable, Callable
import itertools

import networkx as nx

from gufe import SmallMoleculeComponent
from .abstract_network_planner import AbstractRelativeLigandNetworkPlanner

from openfe.setup import LigandNetwork
from openfe.setup.atom_mapping import LigandAtomMapper, LigandAtomMapping


class MaximalNetworkPlanner(AbstractRelativeLigandNetworkPlanner):
    def __init__(
        self,
        mappers: Iterable[LigandAtomMapper],
        mapping_scorer=None,
        # allow_disconnected=True
    ):
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

    def __call__(
        self,
        ligands: Iterable[SmallMoleculeComponent],
    ) -> LigandNetwork:
        """Create a network with all possible proposed mappings.

        This will attempt to create (and optionally score) all possible mappings
        (up to $N(N-1)/2$ for each mapper given). There may be fewer actual
        mappings that this because, when a mapper cannot return a mapping for a
        given pair, there is simply no suggested mapping for that pair.
        This network is typically used as the starting point for other network
        generators (which then optimize based on the scores) or to debug atom
        mappers (to see which mappings the mapper fails to generate).


        Parameters
        ----------
        ligands : Iterable[SmallMoleculeComponent]
          the ligands to include in the LigandNetwork

        """
        nodes = list(ligands)

        mapping_generator = itertools.chain.from_iterable(
            mapper.suggest_mappings(molA, molB)
            for molA, molB in itertools.combinations(nodes, 2)
            for mapper in self._mappers
        )
        if self._mapping_scorer is not None:
            self._mappings = [
                mapping.with_annotations({"score": self._mapping_scorer(mapping)})
                for mapping in mapping_generator
            ]
        else:
            self._mappings = list(mapping_generator)

        network = LigandNetwork(self.mappings, nodes=nodes)
        return network
