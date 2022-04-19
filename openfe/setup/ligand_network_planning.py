# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
import math
from typing import Iterable
import itertools

from openfe.setup import Network, SmallMoleculeComponent, LigandAtomMapper


def generate_radial_network(ligands: Iterable[SmallMoleculeComponent],
                            central_ligand: SmallMoleculeComponent,
                            mappers: Iterable[LigandAtomMapper], scorer=None):
    """Generate a radial network with all ligands connected to a central node

    Also known as hub and spoke or star-map, this plans a Network where
    all ligands are connected via a central ligand.

    Parameters
    ----------
    ligands : iterable of SmallMoleculeComponents
      the ligands to arrange around the central ligand
    central_ligand : SmallMoleculeComponent
      the ligand to use as the hub/central ligand
    mappers : iterable of LigandAtomMappers
      mappers to use, at least 1 required
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
    network : Network
      will have an edge between each ligand and the central ligand, with the
      mapping being the best possible mapping found using the supplied atom
      mappers.
      If no scorer is supplied, the first mapping provided by the iterable
      of mappers will be used.
    """
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

    return Network(edges)


def minimal_spanning_graph(ligands: Iterable[SmallMoleculeComponent],
                           mappers: Iterable[LigandAtomMapper], scorer=None):
    """Plan a Network which connects all ligands with minimal cost

    Parameters
    ----------
    ligands : Iterable of rdkit Molecules
      the ligands to include in the Network
    mappers : Iterable of AtomMappers
      the AtomMappers to use to propose mappings.  At least 1 required,
      but many can be given, in which case all will be tried to find the
      lowest score edges
    scorer : Scoring function
      any callable which takes an LigandAtomMapping and returns a float
    """
    raise NotImplementedError
