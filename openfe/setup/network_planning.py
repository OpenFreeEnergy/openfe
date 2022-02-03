# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

import math

import networkx as nx
from typing import TypeVar

from . import Network
from ..utils.molhashing import hashmol

NetworkXGraph = TypeVar("NetworkXGraph")


class NetworkPlanner:
    """The preparation of a ligand Network"""
    _graph: NetworkXGraph

    def __init__(self):
        self._graph = nx.Graph()

    def __getitem__(self, key):
        key = (hashmol(key[0]), hashmol(key[1]))

        return self._graph[key]

    def add_edge(self, atommapping):
        """Manually add an edge onto an existing Network

        Parameters
        ----------
        atommapping : AtomMapping
          the mapping between two ligands to add

        Raises
        ------
        ValueError
          if an edge between two ligands already exists
        """
        key = (hashmol(atommapping.mol1), hashmol(atommapping.mol2))

        if key in self._graph:
            raise ValueError("This edge already exists in Network")

        self._graph.add_edge(key, atommapping)

    def get_edges(self):
        """Iterable of edges in this Network"""
        yield from self._graph.edges()

    def get_network(self):
        raise NotImplementedError()
        return Network(self._graph)


def generate_radial_graph(ligands, central_ligand, mappers, scorer=None):
    """Radial Network generator

    Parameters
    ----------
    ligands : list of rdkit Molecules
      the ligands to arrange around the central ligand
    central_ligand : rdkit Molecule
      the ligand to use as the hub/central ligand
    mappers : iterable of AtomMappers
      mappers to use, at least 1 required
    scorer : Scorer to use for multiple mappers, optional
      if multiple Mappers are provided, the scorer discerns which to use

    Returns
    -------
    network : NetworkPlanner
      will have an edge between each ligand and the central ligand, with the
      mapping being the best possible mapping found using the supplied atom
      mappers.

      If scorers are not given, then the first AtomMapper to provide a valid
      mapping will be used.
    """
    n = NetworkPlanner()

    for ligand in ligands:
        best_score = math.inf
        best_mapping = None

        for mapper in mappers:
            for mapping in mapper.suggest_mappings(central_ligand, ligand):
                if not scorer:
                    best_mapping = mapping
                    break

                score = scorer(mapping)

                if score < best_score:
                    best_mapping = mapping
                    best_score = score

        if best_mapping is None:
            raise ValueError("No mapping found!")
        n.add_edge(best_mapping)

    return n


def minimal_spanning_graph(ligands, mappers, scorer):
    """Plan a Network which connects all ligands with minimal cost

    Parameters
    ----------
    ligands : list of rdkit Molecules
      the ligands to include in the Network
    mappers : list of AtomMappers
      the AtomMappers to use to propose mappings.  At least 1 required,
      but many can be given, in which case all will be tried to find the
      lowest score edges
    scorer : Scoring function
      any callable which takes an AtomMapping and returns a float
    """
    raise NotImplementedError
