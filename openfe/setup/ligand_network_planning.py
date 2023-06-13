# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import math
from typing import Iterable, Callable, Optional, Union
import itertools
import functools

import networkx as nx
from tqdm.auto import tqdm

from gufe import SmallMoleculeComponent, AtomMapper
from openfe.setup import LigandNetwork
from openfe.setup.atom_mapping import LigandAtomMapping


def generate_radial_network(ligands: Iterable[SmallMoleculeComponent],
                            central_ligand: SmallMoleculeComponent,
                            mappers: Union[AtomMapper, Iterable[AtomMapper]],
                            scorer=None):
    """Generate a radial network with all ligands connected to a central node

    Also known as hub and spoke or star-map, this plans a LigandNetwork where
    all ligands are connected via a central ligand.

    Parameters
    ----------
    ligands : iterable of SmallMoleculeComponents
      the ligands to arrange around the central ligand
    central_ligand : SmallMoleculeComponent
      the ligand to use as the hub/central ligand
    mappers : AtomMapper or iterable of AtomMappers
      mapper(s) to use, at least 1 required
    scorer : scoring function, optional
      a callable which returns a float for any LigandAtomMapping.  Used to
      assign scores to potential mappings, higher scores indicate worse
      mappings.

    Raises
    ------
    ValueError
      if no mapping between the central ligand and any other ligand can be
      found

    Returns
    -------
    network : LigandNetwork
      will have an edge between each ligand and the central ligand, with the
      mapping being the best possible mapping found using the supplied atom
      mappers.
      If no scorer is supplied, the first mapping provided by the iterable
      of mappers will be used.
    """
    if isinstance(mappers, AtomMapper):
        mappers = [mappers]
    edges = []

    for ligand in ligands:
        best_score = math.inf
        best_mapping = None

        for mapping in itertools.chain.from_iterable(
            mapper.suggest_mappings(central_ligand, ligand)
            for mapper in mappers
        ):
            if not scorer:
                best_mapping = mapping
                break

            score = scorer(mapping)
            mapping = mapping.with_annotations({"score": score})

            if score < best_score:
                best_mapping = mapping
                best_score = score

        if best_mapping is None:
            raise ValueError(f"No mapping found for {ligand}")
        edges.append(best_mapping)

    return LigandNetwork(edges)


def generate_maximal_network(
    ligands: Iterable[SmallMoleculeComponent],
    mappers: Union[AtomMapper, Iterable[AtomMapper]],
    scorer: Optional[Callable[[LigandAtomMapping], float]] = None,
    progress: Union[bool, Callable[[Iterable], Iterable]] = True,
    # allow_disconnected=True
):
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
    mappers : AtomMapper or Iterable[AtomMapper]
      the AtomMapper(s) to use to propose mappings.  At least 1 required,
      but many can be given, in which case all will be tried to find the
      lowest score edges
    scorer : Scoring function
      any callable which takes a LigandAtomMapping and returns a float
    progress : Union[bool, Callable[Iterable], Iterable]
      progress bar: if False, no progress bar will be shown. If True, use a
      tqdm progress bar that only appears after 1.5 seconds. You can also
      provide a custom progress bar wrapper as a callable.
    """
    if isinstance(mappers, AtomMapper):
        mappers = [mappers]

    nodes = list(ligands)

    if progress is True:
        # default is a tqdm progress bar
        total = len(nodes) * (len(nodes) - 1) // 2
        progress = functools.partial(tqdm, total=total, delay=1.5)
    elif progress is False:
        progress = lambda x: x
    # otherwise, it should be a user-defined callable

    mapping_generator = itertools.chain.from_iterable(
        mapper.suggest_mappings(molA, molB)
        for molA, molB in progress(itertools.combinations(nodes, 2))
        for mapper in mappers
    )
    if scorer:
        mappings = [mapping.with_annotations({'score': scorer(mapping)})
                    for mapping in mapping_generator]
    else:
        mappings = list(mapping_generator)

    network = LigandNetwork(mappings, nodes=nodes)
    return network


def generate_minimal_spanning_network(
    ligands: Iterable[SmallMoleculeComponent],
    mappers: Union[AtomMapper, Iterable[AtomMapper]],
    scorer: Callable[[LigandAtomMapping], float],
    progress: Union[bool, Callable[[Iterable], Iterable]] = True,
):
    """Plan a LigandNetwork which connects all ligands with minimal cost

    Parameters
    ----------
    ligands : Iterable[SmallMoleculeComponent]
      the ligands to include in the LigandNetwork
    mappers : AtomMapper or Iterable[AtomMapper]
      the AtomMapper(s) to use to propose mappings.  At least 1 required,
      but many can be given, in which case all will be tried to find the
      lowest score edges
    scorer : Scoring function
      any callable which takes a LigandAtomMapping and returns a float
    progress : Union[bool, Callable[Iterable], Iterable]
      progress bar: if False, no progress bar will be shown. If True, use a
      tqdm progress bar that only appears after 1.5 seconds. You can also
      provide a custom progress bar wrapper as a callable.
    """
    if isinstance(mappers, AtomMapper):
        mappers = [mappers]

    # First create a network with all the proposed mappings (scored)
    network = generate_maximal_network(ligands, mappers, scorer, progress)

    # Next analyze that network to create minimal spanning network. Because
    # we carry the original (directed) LigandAtomMapping, we don't lose
    # direction information when converting to an undirected graph.
    min_edges = nx.minimum_spanning_edges(nx.MultiGraph(network.graph),
                                          weight='score')
    min_mappings = [edge_data['object'] for _, _, _, edge_data in min_edges]
    min_network = LigandNetwork(min_mappings)
    missing_nodes = set(network.nodes) - set(min_network.nodes)
    if missing_nodes:
        raise RuntimeError("Unable to create edges to some nodes: "
                           + str(list(missing_nodes)))

    return min_network
