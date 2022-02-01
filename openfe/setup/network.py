import math
import networkx as nx
from rdkit import Chem
from typing import TypeVar


NetworkXGraph = TypeVar("NetworkXGraph")


def _hashmol(mol):
    # canonical by default
    # TODO: coordinates
    return Chem.MolToSmiles(mol)


class Network:
    """A set of ligands and mappings between them"""
    _graph: NetworkXGraph

    def __init__(self):
        self._graph = nx.Graph()

    def __getitem__(self, key):
        key = (_hashmol(key[0]), _hashmol(key[1]))

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
        key = (_hashmol(atommapping.mol1), _hashmol(atommapping.mol2))

        if key in self._graph:
            raise ValueError("This edge already exists in Network")

        self._graph.add_edge(key, atommapping)

    def get_edges(self):
        """Iterable of edges in this Network"""
        yield from self._graph.edges()

    @classmethod
    def from_lomap(cls, lomap_file):
        raise NotImplementedError  # issue #16

    def to_json(self):
        """Return json string representation of this Network"""
        raise NotImplementedError  # issue #15


def generate_radial_graph(ligands, central_ligand, mappers, scorers=None):
    """Radial Network generator

    Parameters
    ----------
    ligands : list of rdkit Molecules
      the ligands to arrange around the central ligand
    central_ligand : rdkit Molecule
      the ligand to use as the hub/central ligand
    mappers : iterable of AtomMappers
      mappers to use, at least 1 required
    scorers : iterable of Scorers, optional
      extra ways to assign scores

    Returns
    -------
    network : Network
      will have an edge between each ligand and the central ligand, with the
      mapping being the best possible mapping found using the supplied atom
      mappers.

      If scorers are not given, then the first AtomMapper to provide a valid
      mapping will be used.
    """
    n = Network()

    for ligand in ligands:
        best_score = math.inf
        best_mapping = None

        for mapper in mappers:
            for mapping in mapper.suggest_mappings(central_ligand, ligand):
                if not scorers:
                    best_mapping = mapping
                    break

                score = sum(scorer(mapping) for scorer in scorers)

                if score < best_score:
                    best_mapping = mapping
                    best_score = score

        if best_mapping is None:
            raise ValueError("No mapping found!")
        n.add_edge(best_mapping)

    return n


def minimal_spanning_graph(ligands, mappers, scorers, weights=None):
    """Plan a Network which connects all ligands with minimal cost

    Parameters
    ----------
    ligands : list of rdkit Molecules
      the ligands to include in the Network
    mappers : list of AtomMappers
      the AtomMappers to use to propose mappings.  At least 1 required,
      but many can be given, in which case all will be tried to find the
      lowest score edges
    scorers : list of Scoring functions
      used to estimate cost to a given AtomMapping
    weights : ???
      used to balance weights of scorers?
    """
    raise NotImplementedError
