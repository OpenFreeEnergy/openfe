# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import math
from pathlib import Path
from typing import Iterable, Callable, Optional, Union
import itertools
from collections import Counter
import functools
import warnings

import networkx as nx
from tqdm.auto import tqdm

from gufe import SmallMoleculeComponent, AtomMapper
from openfe.setup import LigandNetwork
from openfe.setup.atom_mapping import LigandAtomMapping

from lomap import generate_lomap_network, LomapAtomMapper
from lomap.dbmol import _find_common_core


def _hasten_lomap(mapper, ligands):
    """take a mapper and some ligands, put a common core arg into the mapper """
    if mapper.seed:
        return mapper

    try:
        core = _find_common_core([m.to_rdkit() for m in ligands],
                                 element_change=mapper.element_change)
    except RuntimeError:  # in case MCS throws a hissy fit
        core = ""

    return LomapAtomMapper(
        time=mapper.time, threed=mapper.threed, max3d=mapper.max3d,
        element_change=mapper.element_change, seed=core,
        shift=mapper.shift
    )


def generate_radial_network(
    ligands: Iterable[SmallMoleculeComponent],
    central_ligand: SmallMoleculeComponent,
    mappers: Union[AtomMapper, Iterable[AtomMapper]],
    scorer: Optional[Callable[[LigandAtomMapping], float]] = None,
) -> LigandNetwork:
    """
    Plan a radial network with all ligands connected to a central node.

    Also known as hub and spoke or star-map, this plans a LigandNetwork where
    all ligands are connected via a central ligand.

    Parameters
    ----------
    ligands : iterable of SmallMoleculeComponents
      the ligands to arrange around the central ligand.  If the central ligand
      is present it will be ignored (i.e. avoiding a self edge)
    central_ligand : SmallMoleculeComponent
      the ligand to use as the hub/central ligand
    mappers : AtomMapper or iterable of AtomMappers
      mapper(s) to use, at least 1 required
    scorer : scoring function, optional
      a callable which returns a float for any LigandAtomMapping.  Used to
      assign scores to potential mappings; higher scores indicate better
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
    mappers = [_hasten_lomap(m, ligands) if isinstance(m, LomapAtomMapper)
               else m for m in mappers]

    edges = []

    for ligand in ligands:
        if ligand == central_ligand:
            wmsg = (f"The central_ligand {ligand.name} was also found in "
                    "the list of ligands to arrange around the "
                    "central_ligand this will be ignored.")
            warnings.warn(wmsg)
            continue
        best_score = 0.0
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

            if score > best_score:
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
) -> LigandNetwork:
    """
    Plan a network with all possible proposed mappings.

    This will attempt to create (and optionally score) all possible mappings
    (up to :math:`N(N-1)/2` for each mapper given). There may be fewer actual
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
      but many can be given.
    scorer : Scoring function
      any callable which takes a LigandAtomMapping and returns a float
    progress : Union[bool, Callable[Iterable], Iterable]
      progress bar: if False, no progress bar will be shown. If True, use a
      tqdm progress bar that only appears after 1.5 seconds. You can also
      provide a custom progress bar wrapper as a callable.
    """
    if isinstance(mappers, AtomMapper):
        mappers = [mappers]
    mappers = [_hasten_lomap(m, ligands) if isinstance(m, LomapAtomMapper)
               else m for m in mappers]

    nodes = list(ligands)

    if progress is True:
        # default is a tqdm progress bar
        total = len(nodes) * (len(nodes) - 1) // 2
        progress = functools.partial(tqdm, total=total, delay=1.5)
    elif progress is False:
        def progress(x): return x
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
) -> LigandNetwork:
    """
    Plan a network with as few edges as possible with maximum total score

    Parameters
    ----------
    ligands : Iterable[SmallMoleculeComponent]
      the ligands to include in the LigandNetwork
    mappers : AtomMapper or Iterable[AtomMapper]
      the AtomMapper(s) to use to propose mappings.  At least 1 required,
      but many can be given, in which case all will be tried to find the
      highest score edges
    scorer : Scoring function
      any callable which takes a LigandAtomMapping and returns a float
    progress : Union[bool, Callable[Iterable], Iterable]
      progress bar: if False, no progress bar will be shown. If True, use a
      tqdm progress bar that only appears after 1.5 seconds. You can also
      provide a custom progress bar wrapper as a callable.
    """
    if isinstance(mappers, AtomMapper):
        mappers = [mappers]
    mappers = [_hasten_lomap(m, ligands) if isinstance(m, LomapAtomMapper)
               else m for m in mappers]

    # First create a network with all the proposed mappings (scored)
    network = generate_maximal_network(ligands, mappers, scorer, progress)

    # Flip network scores so we can use minimal algorithm
    g2 = nx.MultiGraph()
    for e1, e2, d in network.graph.edges(data=True):
        g2.add_edge(e1, e2, weight=-d['score'], object=d['object'])

    # Next analyze that network to create minimal spanning network. Because
    # we carry the original (directed) LigandAtomMapping, we don't lose
    # direction information when converting to an undirected graph.
    min_edges = nx.minimum_spanning_edges(g2)
    min_mappings = [edge_data['object'] for _, _, _, edge_data in min_edges]
    min_network = LigandNetwork(min_mappings)
    missing_nodes = set(network.nodes) - set(min_network.nodes)
    if missing_nodes:
        raise RuntimeError("Unable to create edges to some nodes: "
                           f"{list(missing_nodes)}")

    return min_network


def generate_minimal_redundant_network(
    ligands: Iterable[SmallMoleculeComponent],
    mappers: Union[AtomMapper, Iterable[AtomMapper]],
    scorer: Callable[[LigandAtomMapping], float],
    progress: Union[bool, Callable[[Iterable], Iterable]] = True,
    mst_num: int = 2,
) -> LigandNetwork:
    """
    Plan a network with a specified amount of redundancy for each node

    Creates a network with as few edges as possible with maximum total score,
    ensuring that every node is connected to two edges to introduce
    statistical redundancy.

    Parameters
    ----------
    ligands : Iterable[SmallMoleculeComponent]
      the ligands to include in the LigandNetwork
    mappers : AtomMapper or Iterable[AtomMapper]
      the AtomMapper(s) to use to propose mappings.  At least 1 required,
      but many can be given, in which case all will be tried to find the
      highest score edges
    scorer : Scoring function
      any callable which takes a LigandAtomMapping and returns a float
    progress : Union[bool, Callable[Iterable], Iterable]
      progress bar: if False, no progress bar will be shown. If True, use a
      tqdm progress bar that only appears after 1.5 seconds. You can also
      provide a custom progress bar wrapper as a callable.
    mst_num: int
      Minimum Spanning Tree number: the number of minimum spanning trees to
      generate. If two, the second-best edges are included in the returned
      network. If three, the third-best edges are also included, etc.
    """
    if isinstance(mappers, AtomMapper):
        mappers = [mappers]
    mappers = [_hasten_lomap(m, ligands) if isinstance(m, LomapAtomMapper)
               else m for m in mappers]

    # First create a network with all the proposed mappings (scored)
    network = generate_maximal_network(ligands, mappers, scorer, progress)

    # Flip network scores so we can use minimal algorithm
    g2 = nx.MultiGraph()
    for e1, e2, d in network.graph.edges(data=True):
        g2.add_edge(e1, e2, weight=-d['score'], object=d['object'])

    # As in .generate_minimal_spanning_network(), use nx to get the minimal
    # network. But now also remove those edges from the fully-connected
    # network, then get the minimal network again. Add mappings from all
    # minimal networks together.
    mappings = []
    for _ in range(mst_num):  # can increase range here for more redundancy
        # get list from generator so that we don't adjust network by calling it:
        current_best_edges = list(nx.minimum_spanning_edges(g2))

        g2.remove_edges_from(current_best_edges)
        for _, _, _, edge_data in current_best_edges:
            mappings.append(edge_data['object'])

    redund_network = LigandNetwork(mappings)
    missing_nodes = set(network.nodes) - set(redund_network.nodes)
    if missing_nodes:
        raise RuntimeError("Unable to create edges to some nodes: "
                           f"{list(missing_nodes)}")

    return redund_network


def generate_network_from_names(
        ligands: list[SmallMoleculeComponent],
        mapper: AtomMapper,
        names: list[tuple[str, str]],
) -> LigandNetwork:
    """
    Generate a :class:`.LigandNetwork` by specifying edges as tuples of names.

    Parameters
    ----------
    ligands : list of SmallMoleculeComponent
      the small molecules to place into the network
    mapper: AtomMapper
      the atom mapper to use to construct edges
    names : list of tuples of names
      the edges to form where the values refer to names of the small molecules,
      eg `[('benzene', 'toluene'), ...]` will create an edge between the
      molecule with names 'benzene' and 'toluene'

    Returns
    -------
    LigandNetwork

    Raises
    ------
    KeyError
      if an invalid name is requested
    ValueError
      if multiple molecules have the same name (this would otherwise be
      problematic)
    """
    nm2idx = {l.name: i for i, l in enumerate(ligands)}

    if len(nm2idx) < len(ligands):
        dupes = Counter((l.name for l in ligands))
        dupe_names = [k for k, v in dupes.items() if v > 1]
        raise ValueError(f"Duplicate names: {dupe_names}")

    try:
        ids = [(nm2idx[nm1], nm2idx[nm2]) for nm1, nm2 in names]
    except KeyError:
        badnames = [nm for nm in itertools.chain.from_iterable(names)
                    if nm not in nm2idx]
        available = [ligand.name for ligand in ligands]
        raise KeyError(f"Invalid name(s) requested {badnames}.  "
                       f"Available: {available}")

    return generate_network_from_indices(ligands, mapper, ids)


def generate_network_from_indices(
        ligands: list[SmallMoleculeComponent],
        mapper: AtomMapper,
        indices: list[tuple[int, int]],
) -> LigandNetwork:
    """
    Generate a :class:`.LigandNetwork` by specifying edges as tuples of indices.

    Parameters
    ----------
    ligands : list of SmallMoleculeComponent
      the small molecules to place into the network
    mapper: AtomMapper
      the atom mapper to use to construct edges
    indices : list of tuples of indices
      the edges to form where the values refer to names of the small molecules,
      eg `[(3, 4), ...]` will create an edge between the 3rd and 4th molecules
      remembering that Python uses 0-based indexing

    Returns
    -------
    LigandNetwork

    Raises
    ------
    IndexError
      if an invalid ligand index is requested
    """
    edges = []

    for i, j in indices:
        try:
            m1, m2 = ligands[i], ligands[j]
        except IndexError:
            raise IndexError(f"Invalid ligand id, requested {i} {j} "
                             f"with {len(ligands)} available")

        mapping = next(mapper.suggest_mappings(m1, m2))

        edges.append(mapping)

    network = LigandNetwork(edges=edges, nodes=ligands)

    if not network.is_connected():
        warnings.warn("Generated network is not fully connected")

    return network


def load_orion_network(
        ligands: list[SmallMoleculeComponent],
        mapper: AtomMapper,
        network_file: Union[str, Path],
) -> LigandNetwork:
    """Load a :class:`.LigandNetwork` from an Orion NES network file.

    Parameters
    ----------
    ligands : list of SmallMoleculeComponent
      the small molecules to place into the network
    mapper: AtomMapper
      the atom mapper to use to construct edges
    network_file : str
      path to NES network file.

    Returns
    -------
    LigandNetwork

    Raises
    ------
    KeyError
      If an unexpected line format is encountered.
    """

    with open(network_file, 'r') as f:
        network_lines = [l.strip().split(' ') for l in f
                         if not l.startswith('#')]

    names = []
    for entry in network_lines:
        if len(entry) != 3 or entry[1] != ">>":
            errmsg = ("line does not match expected name >> name format: "
                      f"{entry}")
            raise KeyError(errmsg)

        names.append((entry[0], entry[2]))

    return generate_network_from_names(ligands, mapper, names)


def load_fepplus_network(
        ligands: list[SmallMoleculeComponent],
        mapper: AtomMapper,
        network_file: Union[str, Path],
) -> LigandNetwork:
    """Load a :class:`.LigandNetwork` from an FEP+ edges network file.

    Parameters
    ----------
    ligands : list of SmallMoleculeComponent
      the small molecules to place into the network
    mapper: AtomMapper
      the atom mapper to use to construct edges
    network_file : str
      path to edges network file.

    Returns
    -------
    LigandNetwork

    Raises
    ------
    KeyError
      If an unexpected line format is encountered.
    """

    with open(network_file, 'r') as f:
        network_lines = [l.split() for l in f.readlines()]

    names = []
    for entry in network_lines:
        if len(entry) != 5 or entry[1] != '#' or entry[3] != '->':
            errmsg = ("line does not match expected format "
                      f"hash:hash # name -> name\n"
                      "line format: {entry}")
            raise KeyError(errmsg)

        names.append((entry[2], entry[4]))

    return generate_network_from_names(ligands, mapper, names)
