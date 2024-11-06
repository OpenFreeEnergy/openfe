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
from konnektor.network_planners import (
    StarNetworkGenerator,
    MaximalNetworkGenerator,
    RedundantMinimalSpanningTreeNetworkGenerator,
    MinimalSpanningTreeNetworkGenerator,
    ExplicitNetworkGenerator,
)
from konnektor import network_analysis, network_planners, network_tools


def _hasten_lomap(mapper, ligands):
    """take a mapper and some ligands, put a common core arg into the mapper"""
    if mapper.seed:
        return mapper

    try:
        core = _find_common_core(
            [m.to_rdkit() for m in ligands],
            element_change=mapper.element_change,
        )
    except RuntimeError:  # in case MCS throws a hissy fit
        core = ""

    return LomapAtomMapper(
        time=mapper.time,
        threed=mapper.threed,
        max3d=mapper.max3d,
        element_change=mapper.element_change,
        seed=core,
        shift=mapper.shift,
    )


def generate_radial_network(
    ligands: Iterable[SmallMoleculeComponent],
    mappers: Union[AtomMapper, Iterable[AtomMapper]],
    central_ligand: Union[SmallMoleculeComponent, str, int, None],
    scorer: Optional[Callable[[LigandAtomMapping], float]] = None,
    n_processes: int = 1,
    progress: bool = False,
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
    mappers : AtomMapper or iterable of AtomMappers
      mapper(s) to use, at least 1 required
    central_ligand : SmallMoleculeComponent or str or int
      the ligand to use as the hub/central ligand.
      If this is a string, this should match to one and only one ligand name.
      If this is an integer, this refers to the index from within ligands
    scorer : scoring function, optional
      a callable which returns a float for any LigandAtomMapping.  Used to
      assign scores to potential mappings; higher scores indicate better
      mappings.
    progress : Union[bool, Callable[Iterable], Iterable]
      progress bar: if False, no progress bar will be shown. If True, use a
      tqdm progress bar that only appears after 1.5 seconds. You can also
      provide a custom progress bar wrapper as a callable.
    n_processes: int
        number of cpu processes to use if parallelizing network generation.

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
    mappers = [
        _hasten_lomap(m, ligands) if isinstance(m, LomapAtomMapper) else m
        for m in mappers
    ]

    # handle central_ligand arg possibilities
    # after this, central_ligand is resolved to a SmallMoleculeComponent
    if isinstance(central_ligand, int):
        ligands = list(ligands)
        try:
            central_ligand = ligands[central_ligand]
            ligands.remove(central_ligand)
        except IndexError:
            raise ValueError(
                f"index '{central_ligand}' out of bounds, there are "
                f"{len(ligands)} ligands"
            )
    elif isinstance(central_ligand, str):
        ligands = list(ligands)
        possibles = [l for l in ligands if l.name == central_ligand]
        if not possibles:
            raise ValueError(
                f"No ligand called '{central_ligand}' "
                f"available: {', '.join(l.name for l in ligands)}"
            )
        if len(possibles) > 1:
            raise ValueError(f"Multiple ligands called '{central_ligand}'")
        central_ligand = possibles[0]
        ligands.remove(central_ligand)

    # Construct network
    network_planner = StarNetworkGenerator(
        mappers=mappers,
        scorer=scorer,
        progress=progress,
        n_processes=n_processes,
    )

    network = network_planner.generate_ligand_network(
        components=ligands, central_component=central_ligand
    )

    return network


def generate_maximal_network(
    ligands: Iterable[SmallMoleculeComponent],
    mappers: Union[AtomMapper, Iterable[AtomMapper]],
    scorer: Optional[Callable[[LigandAtomMapping], float]] = None,
    progress: Union[bool, Callable[[Iterable], Iterable]] = True,
    n_processes: int = 1,
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
    mapper : AtomMapper or Iterable[AtomMapper]
      the AtomMapper(s) to use to propose mappings.  At least 1 required,
      but many can be given.
    scorer : Scoring function
      any callable which takes a LigandAtomMapping and returns a float
    progress : Union[bool, Callable[Iterable], Iterable]
      progress bar: if False, no progress bar will be shown. If True, use a
      tqdm progress bar that only appears after 1.5 seconds. You can also
      provide a custom progress bar wrapper as a callable.
    n_processes: int
        number of cpu processes to use if parallelizing network generation.
    """
    if isinstance(mappers, AtomMapper):
        mappers = [mappers]
    mappers = [
        _hasten_lomap(m, ligands) if isinstance(m, LomapAtomMapper) else m
        for m in mappers
    ]
    nodes = list(ligands)

    # Construct network
    network_planner = MaximalNetworkGenerator(
        mappers=mappers,
        scorer=scorer,
        progress=progress,
        n_processes=n_processes,
    )

    network = network_planner.generate_ligand_network(nodes)

    return network


def generate_minimal_spanning_network(
    ligands: Iterable[SmallMoleculeComponent],
    mappers: Union[AtomMapper, Iterable[AtomMapper]],
    scorer: Callable[[LigandAtomMapping], float],
    progress: Union[bool, Callable[[Iterable], Iterable]] = True,
    n_processes: int = 1,
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
    n_processes: int
        number of cpu processes to use if parallelizing network generation.
    """
    if isinstance(mappers, AtomMapper):
        mappers = [mappers]
    mappers = [
        _hasten_lomap(m, ligands) if isinstance(m, LomapAtomMapper) else m
        for m in mappers
    ]
    nodes = list(ligands)

    # Construct network
    network_planner = MinimalSpanningTreeNetworkGenerator(
        mappers=mappers,
        scorer=scorer,
        progress=progress,
        n_processes=n_processes,
    )

    network = network_planner.generate_ligand_network(nodes)

    return network


def generate_minimal_redundant_network(
    ligands: Iterable[SmallMoleculeComponent],
    mappers: Union[AtomMapper, Iterable[AtomMapper]],
    scorer: Callable[[LigandAtomMapping], float],
    progress: Union[bool, Callable[[Iterable], Iterable]] = True,
    mst_num: int = 2,
    n_processes: int = 1,
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
    mst_num : int
      Minimum Spanning Tree number: the number of minimum spanning trees to
      generate. If two, the second-best edges are included in the returned
      network. If three, the third-best edges are also included, etc.
    n_processes: int
        number of threads to use if parallelizing network generation

    """
    if isinstance(mappers, AtomMapper):
        mappers = [mappers]
    mappers = [
        _hasten_lomap(m, ligands) if isinstance(m, LomapAtomMapper) else m
        for m in mappers
    ]
    nodes = list(ligands)

    # Construct network
    network_planner = RedundantMinimalSpanningTreeNetworkGenerator(
        mappers=mappers,
        scorer=scorer,
        progress=progress,
        n_redundancy=mst_num,
        n_processes=n_processes,
    )

    network = network_planner.generate_ligand_network(nodes)

    return network


def generate_network_from_names(
    ligands: list[SmallMoleculeComponent],
    mappers: Union[AtomMapper, Iterable[AtomMapper]],
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
    nodes = list(ligands)

    network_planner = ExplicitNetworkGenerator(mappers=mappers, scorer=None)

    network = network_planner.generate_network_from_names(
        ligands=nodes, names=names
    )

    return network


def generate_network_from_indices(
    ligands: list[SmallMoleculeComponent],
    mappers: Union[AtomMapper, Iterable[AtomMapper]],
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
    nodes = list(ligands)

    network_planner = ExplicitNetworkGenerator(mappers=mappers, scorer=None)
    network = network_planner.generate_network_from_indices(
        ligands=nodes, indices=indices
    )
    return network


def load_orion_network(
    ligands: list[SmallMoleculeComponent],
    mappers: Union[AtomMapper, Iterable[AtomMapper]],
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

    with open(network_file, "r") as f:
        network_lines = [
            line.strip().split(" ") for line in f if not line.startswith("#")
        ]

    names = []
    for entry in network_lines:
        if len(entry) != 3 or entry[1] != ">>":
            errmsg = (
                "line does not match expected name >> name format: " f"{entry}"
            )
            raise KeyError(errmsg)

        names.append((entry[0], entry[2]))

    network_planner = ExplicitNetworkGenerator(mappers=mappers, scorer=None)
    network = network_planner.generate_network_from_names(
        ligands=ligands, names=names
    )

    return network


def load_fepplus_network(
    ligands: list[SmallMoleculeComponent],
    mappers: Union[AtomMapper, Iterable[AtomMapper]],
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

    with open(network_file, "r") as f:
        network_lines = [l.split() for l in f.readlines()]

    names = []
    for entry in network_lines:
        if len(entry) != 5 or entry[1] != "#" or entry[3] != "->":
            errmsg = (
                "line does not match expected format "
                f"hash:hash # name -> name\n"
                "line format: {entry}"
            )
            raise KeyError(errmsg)

        names.append((entry[2], entry[4]))

    network_planner = ExplicitNetworkGenerator(mappers=mappers, scorer=None)
    network = network_planner.generate_network_from_names(
        ligands=ligands, names=names
    )
    return network
